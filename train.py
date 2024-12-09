import os
import datetime
import json
import time
import argparse
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoConfig
from utils import get_mfu, set_all_seed, to_readable_format, save_checkpoint, load_checkpoint
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from model import Llama
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer
from functools import partial
import numpy as np
from datasets import Features, Sequence, Value
import wandb


class MicroBatchDataLoader(DataLoader):
    def __init__(self,  micro_batch_size, seq_length, dataset_name, tokenizer_name, num_workers, num_proc, grad_acc_steps, split="train"):
        self.micro_batch_size = micro_batch_size
        self.seq_length = seq_length
        self.grad_acc_steps = grad_acc_steps
        self.total_batch_size = micro_batch_size * grad_acc_steps
        self.tokens_per_step = self.total_batch_size * self.seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset = load_dataset(dataset_name, split=split)
        
        # Tokenize and chunk the dataset
        self.tokenized_dataset = self.tokenize_dataset(self.dataset, "text", self.seq_length, num_proc)
        
        super().__init__(
            self.tokenized_dataset,
            batch_size=micro_batch_size,
            collate_fn=self.collate_batch, 
            pin_memory=True, 
            num_workers=num_workers, 
            shuffle=False
        )

    @staticmethod
    def tokenizer_group_text(examples, tokenizer, sequence_length):
        """Tokenize a list of texts and group them in chunks of sequence_length + 1"""
        tokenized_text_batch = tokenizer.batch_encode_plus(
            examples,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors='np'
        )
        concatenated_tokens = {'input_ids': np.concatenate(tokenized_text_batch['input_ids'])}
        total_length = len(concatenated_tokens['input_ids'])
        if total_length >= sequence_length + 1:
            total_length = ((total_length - 1) // sequence_length) * sequence_length + 1
        result = {
            'input_ids': [
                concatenated_tokens['input_ids'][i : i + sequence_length + 1]
                for i in range(0, total_length - sequence_length, sequence_length)
            ]
        }
        return result

    def tokenize_dataset(self, dataset, text_column_name, sequence_length, num_proc):
        """Tokenize the dataset and group texts in chunks of sequence_length + 1"""
        # Create a partial function with fixed arguments
        tokenizer_func = partial(
            self.tokenizer_group_text,
            tokenizer=self.tokenizer,
            sequence_length=sequence_length
        )

        tokenized_dataset = dataset.map(
            tokenizer_func,
            input_columns=text_column_name,
            remove_columns=dataset.column_names,
            features=Features({
                "input_ids": Sequence(feature=Value(dtype="int64"), length=sequence_length + 1)
            }),
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {sequence_length+1}",
        )

        return tokenized_dataset

    def collate_batch(self, batch):
        batch_input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
        batch_size = batch_input_ids.size(0)
        input_ids = batch_input_ids[:, :self.seq_length].contiguous()
        target_ids = batch_input_ids[:, 1:self.seq_length+1].contiguous()
        position_ids = torch.arange(0, self.seq_length, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).contiguous() 
        local_attn_mask = torch.tril(torch.ones((self.seq_length, self.seq_length), dtype=torch.bool))
        attn_mask = local_attn_mask.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "position_ids": position_ids,
            "attn_mask": attn_mask,
            "hidden_states": None
        }
    
    def __iter__(self):
        if self._iterator is None:
            self._iterator = super().__iter__()
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = super().__iter__()
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = None
            raise StopIteration
        return batch

def train_step(model, data_loader, device):
    acc_loss = 0.0
    
    for i in range(data_loader.grad_acc_steps):
        # get the next batch
        batch = next(data_loader)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        outputs = model(input_ids=input_ids)

        # compute the loss
        batch_size, seq_len = input_ids.shape
        target_ids = target_ids.reshape(-1)
        outputs = outputs.view(seq_len*batch_size, -1)
        loss = F.cross_entropy(outputs, target_ids, reduction='mean') / data_loader.grad_acc_steps
        
        loss.backward()

        acc_loss += loss.item()

    return acc_loss

if __name__ == "__main__":        
    os.environ["TOKENIZERS_PARALLELISM"] = '4'
    os.environ["DEVICE"] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32 # if GPU is not available or not supported, use torch.float32
    
    # hyperparameters
    SEQ_LEN = 1024
    MICRO_BATCH_SIZE = 32
    GRAD_ACC_STEPS = 4
    LEARNING_RATE = 3e-4
    NUM_SAMPLES = None
    MAX_TOKENS = 1e12
    SEED = 42
    TOTAL_TRAIN_STEPS = 1000
    DATASET_NAME = "roneneldan/TinyStories"
    NUM_WORKERS = 1
    NUM_PROC = 16 # multiple process to tokenize the dataset
    USE_WANDB = False
    LOAD_PATH = 'checkpoints/500' # /checkpoints/400
    CHECKPOINT_DIR = "checkpoints"
    CHECKPOINT_FREQ = 100
    MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
    
    backend = "nccl"
    device = torch.device("cuda")

    set_all_seed(SEED)
    
    start_time = time.time()
    data_loader = MicroBatchDataLoader(
        micro_batch_size=MICRO_BATCH_SIZE,
        seq_length=SEQ_LEN,
        dataset_name=DATASET_NAME,
        tokenizer_name=MODEL_NAME,
        grad_acc_steps=GRAD_ACC_STEPS,
        num_workers=NUM_WORKERS,
        num_proc=NUM_PROC,
    )
    print(f"init dataloader time: {time.time()-start_time:.2f}s")
    tokens_per_step = data_loader.total_batch_size * SEQ_LEN
    print("Tokens per step:", to_readable_format(tokens_per_step))

    if USE_WANDB:
        wandb.init(
            project="Cloud_and_ML_Final_Project",
            name=f"{MODEL_NAME}_{to_readable_format(tokens_per_step)}",
            config={
                "dataset": DATASET_NAME,
                "max_tokens": MAX_TOKENS,
                "learning_rate": LEARNING_RATE,
                "seed": SEED,
                "micro_batch_size": MICRO_BATCH_SIZE,
                "global_batch_size": data_loader.total_batch_size,
                "gradient_accumulation": GRAD_ACC_STEPS,
            },
        )

    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.max_position_embeddings = SEQ_LEN

    start_time = time.time()
    model = Llama(config=model_config)
    print(f"init model time: {time.time()-start_time:.2f}s")

    start_time = time.time()

    model.to(dtype).to(device)
        
    model.train()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {to_readable_format(num_params)}")
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    trained_tokens, step = 0, 0
    if LOAD_PATH:
        step, trained_tokens = load_checkpoint(model, LOAD_PATH, optimizer)
    
    
    while MAX_TOKENS is None or trained_tokens < MAX_TOKENS:
        step_start_time = time.time()
        optimizer.zero_grad()
        
        loss = train_step(model, data_loader, device)
        
        optimizer.step()
        trained_tokens += tokens_per_step
        step += 1

        step_duration = time.time() - step_start_time
        tokens_per_second = tokens_per_step / step_duration
        mfu = get_mfu(tokens_per_second, num_params, model_config)
        
        print(
            f"Step: {step:<5d} | "
            f"Loss: {loss:6.4f} | "
            f"Global batch size: {to_readable_format(tokens_per_step):>7s} | "
            f"Tokens/s: {to_readable_format(tokens_per_second):>7s} | "
            f"Tokens: {to_readable_format(trained_tokens):>7s}{('/' + to_readable_format(MAX_TOKENS)) if MAX_TOKENS else ''} | "
            f"MFU: {mfu:5.2f}% | "
            f"Memory usage: {torch.cuda.memory_reserved() / 1e9:6.2f}GB",
        )
        
        if USE_WANDB:
            wandb.log({"loss": loss, "tokens_per_step": tokens_per_step, "tokens_per_second": tokens_per_step / step_duration,\
                        "mfu": mfu, "tokens_per_second_per_gpu": tokens_per_second_per_gpu, "memory_usage": torch.cuda.memory_reserved() / 1e9, "trained_tokens": trained_tokens})
        
        if step % CHECKPOINT_FREQ == 0:
            save_checkpoint(model, optimizer, step, trained_tokens, CHECKPOINT_DIR+f"/{step}")
        
        if step >= TOTAL_TRAIN_STEPS:
            break
    
    if USE_WANDB:
        wandb.finish()
