import torch
import torch.distributed as dist
from tqdm import tqdm
import src.config as config
from src.distributed_utils import is_main_process

# Existing training function for single GPU
def train_model_dpo_simple(
    dpo_loss_fn, optimizer,
    policy_model, reference_model, train_loader, val_loader,
    num_epochs, eval_freq, eval_iter):
    """
    Fine-tunes the policy model using the DPO method on a single GPU.

    :param dpo_loss_fn: The DPO loss function.
    :param optimizer: The optimizer.
    :param policy_model: The model to be fine-tuned.
    :param reference_model: The reference model (with frozen weights).
    :param train_loader: The DataLoader for the training dataset.
    :param val_loader: The DataLoader for the validation dataset.
    :param num_epochs: The number of training epochs.
    :param eval_freq: The frequency (in steps) at which to perform evaluations.
    :param eval_iter: The number of evaluation iterations.
    :return: A dictionary tracking various losses and reward metrics.
    """
    print("Starting training...")
    print("Ref model grad status:", next(reference_model.parameters()).requires_grad)
    # Initialize lists to track losses and tokens seen
    tracking = {
        "train_losses": [],
        "train_chosen_rewards": [],
        "train_rejected_rewards": [],
        "val_losses": [],
        "val_chosen_rewards": [],
        "val_rejected_rewards": [],
        "tokens_seen": []
    }
    tokens_seen, global_step = 0, -1

    prev_val_loss = float('inf')
    patience = config.early_stopping_patience
    patience_counter = 0
    max_reward_margin = config.max_reward_margin

    # Main training loop
    for epoch in range(num_epochs):
        policy_model.train()  # Set model to training mode

        # Add tqdm progress bar for each epoch
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for batch_idx, batch in enumerate(train_loop):
            optimizer.zero_grad()
            
            loss, chosen_rewards, rejected_rewards = dpo_loss_fn.compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model
            )
            loss.backward()  # Compute gradients

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)  # clipping
            print(f"Step {global_step+1}: Grad norm: {grad_norm.item():.4f}")

            optimizer.step() # Update model parameters

            # Track tokens processed
            tokens_seen += batch["chosen"].numel()
            global_step += 1

            # Update progress bar
            train_loop.set_postfix(loss=f"{loss.item():.4f}", reward_diff=f"{(chosen_rewards-rejected_rewards).item():.4f}")

            # Evaluation step
            if global_step % eval_freq == 0:
                res = dpo_loss_fn.evaluate_dpo_loss_loader(
                    policy_model=policy_model,
                    reference_model=reference_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    eval_iter=eval_iter
                )
                
                tracking["train_losses"].append(res["train_loss"])
                tracking["train_chosen_rewards"].append(res["train_chosen_reward"])
                tracking["train_rejected_rewards"].append(res["train_rejected_reward"])
                tracking["val_losses"].append(res["val_loss"])
                tracking["val_chosen_rewards"].append(res["val_chosen_reward"])
                tracking["val_rejected_rewards"].append(res["val_rejected_reward"])
                tracking["tokens_seen"].append(tokens_seen)
                train_reward_margin = res["train_chosen_reward"] - res["train_rejected_reward"]
                val_reward_margin = res["val_chosen_reward"] - res["val_rejected_reward"]

                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {res['train_loss']:.3f}, Val loss {res['val_loss']:.3f}, "
                    f"Train reward margins {train_reward_margin:.3f}, "
                    f"Val reward margins {val_reward_margin:.3f}"
                )

                # Stop if reward margin is too large
                if train_reward_margin > max_reward_margin or val_reward_margin > max_reward_margin:
                    print(f"Training stopped: Reward margin too large ({train_reward_margin:.2f}, {val_reward_margin:.2f})")
                    break
                    
                # Stop if validation loss starts increasing
                if res["val_loss"] > prev_val_loss:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Training stopped: Early stopping triggered after {patience} epochs of increasing validation loss")
                        break
                else:
                    patience_counter = 0
                    prev_val_loss = res["val_loss"]
    
    print("Training completed.")
    return tracking


# New function for distributed training on multiple GPUs
def train_model_dpo_distributed(
    dpo_loss_fn, optimizer, scheduler,
    policy_model, reference_model, train_loader, val_loader, train_sampler,
    num_epochs, eval_freq, eval_iter, rank, world_size):
    """
    Fine-tunes the policy model using DPO method with distributed training across multiple GPUs.

    :param dpo_loss_fn: The DPO loss function.
    :param optimizer: The optimizer.
    :param scheduler: Learning rate scheduler.
    :param policy_model: The model to be fine-tuned (wrapped in DistributedDataParallel).
    :param reference_model: The reference model (with frozen weights, wrapped in DistributedDataParallel).
    :param train_loader: The distributed DataLoader for the training dataset.
    :param val_loader: The distributed DataLoader for the validation dataset.
    :param train_sampler: DistributedSampler for the training data.
    :param num_epochs: The number of training epochs.
    :param eval_freq: The frequency (in steps) at which to perform evaluations.
    :param eval_iter: The number of evaluation iterations.
    :param rank: The rank of the current process.
    :param world_size: Total number of processes.
    :return: A dictionary tracking various losses and reward metrics.
    """
    # Only print from main process to avoid log clutter
    if is_main_process(rank):
        print("Starting distributed training...")
        print(f"Training on {world_size} GPUs")
        print("Ref model grad status:", next(reference_model.parameters()).requires_grad)
    
    # Initialize tracking metrics
    tracking = {
        "train_losses": [],
        "train_chosen_rewards": [],
        "train_rejected_rewards": [],
        "val_losses": [],
        "val_chosen_rewards": [],
        "val_rejected_rewards": [],
        "tokens_seen": []
    }
    tokens_seen, global_step = 0, -1

    # Early stopping parameters
    prev_val_loss = float('inf')
    patience = config.early_stopping_patience
    patience_counter = 0
    max_reward_margin = config.max_reward_margin
    early_stop = False

    # Main training loop
    for epoch in range(num_epochs):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)
        
        # Set model to training mode
        policy_model.train()
        
        # Create progress bar only on the main process
        if is_main_process(rank):
            train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        else:
            train_loop = train_loader
        
        # Iterate over batches
        for batch_idx, batch in enumerate(train_loop):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass and loss computation
            loss, chosen_rewards, rejected_rewards = dpo_loss_fn.compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Scheduler step if provided
            if scheduler is not None:
                scheduler.step()
            
            # Track tokens seen (only locally)
            local_tokens = batch["chosen"].numel()
            tokens_seen += local_tokens
            global_step += 1
            
            # Gather tokens seen from all processes
            if dist.is_initialized():
                token_tensor = torch.tensor([local_tokens], device=f'cuda:{rank}')
                dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
                global_tokens = token_tensor.item()
            else:
                global_tokens = local_tokens
            
            # Update progress bar on main process
            if is_main_process(rank):
                train_loop.set_postfix(
                    loss=f"{loss.item():.4f}", 
                    reward_diff=f"{(chosen_rewards-rejected_rewards).item():.4f}",
                    grad_norm=f"{grad_norm.item():.4f}"
                )
            
            # Evaluation step (only main process does evaluation)
            if global_step % eval_freq == 0 and is_main_process(rank):
                res = dpo_loss_fn.evaluate_dpo_loss_loader(
                    policy_model=policy_model,
                    reference_model=reference_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    eval_iter=eval_iter
                )
                
                # Update tracking metrics
                tracking["train_losses"].append(res["train_loss"])
                tracking["train_chosen_rewards"].append(res["train_chosen_reward"])
                tracking["train_rejected_rewards"].append(res["train_rejected_reward"])
                tracking["val_losses"].append(res["val_loss"])
                tracking["val_chosen_rewards"].append(res["val_chosen_reward"])
                tracking["val_rejected_rewards"].append(res["val_rejected_reward"])
                tracking["tokens_seen"].append(tokens_seen)
                
                # Calculate reward margins
                train_reward_margin = res["train_chosen_reward"] - res["train_rejected_reward"]
                val_reward_margin = res["val_chosen_reward"] - res["val_rejected_reward"]
                
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {res['train_loss']:.3f}, Val loss {res['val_loss']:.3f}, "
                    f"Train reward margins {train_reward_margin:.3f}, "
                    f"Val reward margins {val_reward_margin:.3f}"
                )
                
                # Early stopping checks
                # Check if reward margin is too large
                if train_reward_margin > max_reward_margin or val_reward_margin > max_reward_margin:
                    print(f"Training stopped: Reward margin too large ({train_reward_margin:.2f}, {val_reward_margin:.2f})")
                    early_stop = True
                
                # Check if validation loss is increasing
                if res["val_loss"] > prev_val_loss:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Training stopped: Early stopping triggered after {patience} steps of increasing validation loss")
                        early_stop = True
                else:
                    patience_counter = 0
                    prev_val_loss = res["val_loss"]
            
            # Broadcast early_stop flag to all processes
            if dist.is_initialized():
                early_stop_tensor = torch.tensor([1 if early_stop else 0], device=f'cuda:{rank}')
                dist.broadcast(early_stop_tensor, src=0)
                early_stop = bool(early_stop_tensor.item())
            
            # Break from batch loop if early stopping
            if early_stop:
                break
        
        # Break from epoch loop if early stopping
        if early_stop:
            break
    
    # Synchronize processes after training
    if dist.is_initialized():
        dist.barrier()
    
    # Only print from main process
    if is_main_process(rank):
        print("Distributed training completed.")
    
    return tracking
