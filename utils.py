import os
import torch
import random
import numpy as np

def set_all_seed(seed):
    for module in [random, np.random]: module.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    
def to_readable_format(num, precision=2):
    if num >= 1e12:
        return f"{num / 1e12:.{precision}f}T"
    elif num >= 1e9:
        return f"{num / 1e9:.{precision}f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"

def get_mfu(tokens_per_second, num_params, model_config, theoretical_flops = 989.5 * 10 ** 12):
    num_layers = model_config.num_hidden_layers
    hidden_dim = model_config.hidden_size
    seq_len = model_config.max_position_embeddings
    flops_per_token = 6 * num_params + 12 * num_layers * hidden_dim * seq_len
    mfu = tokens_per_second * flops_per_token / theoretical_flops * 100 # percentage
    return mfu

def save_checkpoint(model, optimizer, trained_steps, trained_tokens, out_dir):
    """Save the model/optimizer states/steps to a checkpoint file."""
    ckpt_name = f"weights.pth"
    path = os.path.join(out_dir, ckpt_name)

    os.makedirs(out_dir, exist_ok=True)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'trained_steps': trained_steps,
        'trained_tokens': trained_tokens
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, out_dir, optimizer = None):
    """Load the model/optimizer states from the latest checkpoint. Assume the topology is the same."""
    ckpt_name = f"weights.pth"
    path = os.path.join(out_dir, ckpt_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['trained_steps'], checkpoint['trained_tokens']
