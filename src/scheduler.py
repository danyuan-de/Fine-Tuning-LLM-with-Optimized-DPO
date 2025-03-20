from torch.optim import lr_scheduler
import math

def get_scheduler(optimizer, warmup_steps, total_steps):
    """
    Create a learning rate scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        
    Returns:
        A learning rate scheduler
    """
    def lr_lambda(current_step):
        # Linear warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
    return lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Usage in main.py:
# 
# total_steps = num_epochs * len(train_loader) // gradient_accumulation_steps
# scheduler = get_scheduler(optimizer, warmup_steps=config.warmup_steps, total_steps=total_steps)