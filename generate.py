from model import Llama
from utils import load_checkpoint
import torch
from transformers import AutoConfig, AutoTokenizer
import torch.nn.functional as F


MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
LOAD_PATH = 'checkpoints/1000'
SEQ_LEN = 1024
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_config = AutoConfig.from_pretrained(MODEL_NAME)
model_config.max_position_embeddings = SEQ_LEN

model = Llama(config=model_config).to(device)
trained_steps, trained_tokens = load_checkpoint(model, LOAD_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

test_prompt = "Once upon a time, there was a"

def _generate_one_token(input_ids):
    output_logits = model(input_ids)
    # Get probabilities from logits using softmax
    probs = F.softmax(output_logits[0, -1, :], dim=-1)
    
    # Get top k values and indices
    k = 10  # Can be adjusted for different levels of randomness
    top_k_probs, top_k_indices = torch.topk(probs, k)
    
    # Sample from the top k distribution
    sampled_idx = torch.multinomial(top_k_probs, num_samples=1)
    token = top_k_indices[sampled_idx]
    return token

def generate_text(prompt, max_tokens = 100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    current_ids = inputs.input_ids
    for _ in range(max_tokens):
        token = _generate_one_token(current_ids)
        current_ids = torch.cat([current_ids, token.unsqueeze(0)], dim=-1)
    return tokenizer.decode(current_ids[0], skip_special_tokens=True)
    
if __name__ == "__main__":
    print(generate_text(test_prompt))

