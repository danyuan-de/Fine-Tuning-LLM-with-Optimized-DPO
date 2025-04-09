import os
import torch
import src.config as config
from src.utility import get_output_filename, postprocess_response
# from src.gpuMonitor import log_memory_snapshot
from tqdm import tqdm
import json

# Define the training function
def train_model(
    dpo_loss_fn, optimizer, scheduler,
    policy_model, reference_model, train_loader, val_loader,
    num_epochs, eval_freq, eval_iter, gradient_accumulation_steps=1, 
    log_memory=False, tokenizer=None):
    """
    Fine-tunes the policy model using the DPO method.

    :param policy_model: The model to be fine-tuned.
    :param reference_model: The reference model (with frozen weights).
    :param train_loader: The DataLoader for the training dataset.
    :param val_loader: The DataLoader for the validation dataset.
    :param optimizer: The optimizer.
    :param num_epochs: The number of training epochs.
    :param beta: The beta value used in the DPO loss.
    :param eval_freq: The frequency (in steps) at which to perform evaluations.
    :param eval_iter: The number of evaluation iterations.
    :param tokenizer: The tokenizer to decode batch data (for debugging)
    :return: A dictionary tracking various losses and reward metrics.
    """
    print("Ref model grad status:", next(reference_model.parameters()).requires_grad)
    print("Policy model grad status:", next(policy_model.parameters()).requires_grad)
    # Initialize lists to track losses and tokens seen
    tracking = {
        "train_losses": [],
        "train_chosen_rewards": [],
        "train_rejected_rewards": [],
        "val_losses": [],
        "val_chosen_rewards": [],
        "val_rejected_rewards": [],
        "tokens_seen": [],
        "memory_usage": [] if log_memory else None,
        "batch_records": []  # Add a new field to store batch info
    }
    tokens_seen, global_step = 0, -1
    accumulated_tokens = 0
    prev_loss = float("inf")

    # Main training loop
    for epoch in range(num_epochs):
        # if log_memory:
        #     log_memory_snapshot(f"Starting epoch {epoch+1}/{num_epochs}")
            
        policy_model.train()  # Set model to training mode

        # Zero the gradients at the beginning of each epoch
        optimizer.zero_grad()

        # Add tqdm progress bar for each epoch
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        try:
            for batch_idx, batch in enumerate(train_loop):
                # if log_memory and batch_idx % max(1, len(train_loader) // 10) == 0:
                #     log_memory_snapshot(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}")

                loss, chosen_rewards, rejected_rewards = dpo_loss_fn.compute_dpo_loss_batch(
                        batch=batch,
                        policy_model=policy_model,
                        reference_model=reference_model
                )
                loss = loss / gradient_accumulation_steps  # Scale loss for gradient accumulation
                loss.backward()  # Compute gradients

                # Check if loss changed significantly (increased or decreased)
                loss_value = loss.item() * gradient_accumulation_steps  # Rescale for reporting
                loss_change = loss_value - prev_loss
                
                # Log batch data if there's a significant change in loss
                if abs(loss_change) > 0.1 and tokenizer is not None:  # Threshold for significant change
                    # Sample an item from the batch to log
                    sample_idx = 0
                    
                    # Try to decode data - protect with try/except
                    try:
                        # Extract just the response part using the postprocess_response function
                        chosen_text = postprocess_response(tokenizer.decode(batch["chosen"][sample_idx]))
                        rejected_text = postprocess_response(tokenizer.decode(batch["rejected"][sample_idx]))
                        
                        # Truncate very long texts to first 200 chars to keep logs manageable
                        if len(chosen_text) > 200:
                            chosen_text = chosen_text[:200] + "..."
                        if len(rejected_text) > 200:
                            rejected_text = rejected_text[:200] + "..."
                        
                        # Record batch info with loss change
                        batch_record = {
                            "step": global_step + 1,
                            "batch_idx": batch_idx,
                            "loss": loss_value,
                            "loss_change": loss_change,
                            "reward_diff": chosen_rewards.item() - rejected_rewards.item(),
                            "sample_chosen": chosen_text,
                            "sample_rejected": rejected_text
                        }
                        
                        tracking["batch_records"].append(batch_record)
                        
                        # Log to console as well
                        print(f"\n{'=='*40}\nLoss Change: {loss_change:.4f} at batch {batch_idx}")
                        print(f"New loss: {loss_value:.4f}, Previous loss: {prev_loss:.4f}")
                        print(f"Chosen sample: {chosen_text[:100]}...")
                        print(f"Rejected sample: {rejected_text[:100]}...")
                        print(f"{'=='*40}\n")
                        
                    except Exception as e:
                        print(f"Error decoding batch: {e}")
                
                # Update previous loss
                prev_loss = loss_value

                # Update step info for progress bar
                reward_diff = chosen_rewards.item() - rejected_rewards.item()
                train_loop.set_postfix(
                    loss=f"{loss.item() * gradient_accumulation_steps:.4f}", 
                    reward_diff=f"{reward_diff:.4f}",
                    step=f"{(batch_idx % gradient_accumulation_steps) + 1}/{gradient_accumulation_steps}"
                )

                # Only perform optimizer step after accumulating gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                    # Gradient clipping, monitor if learning rate is appropriate or if the model is experiencing gradient issues.
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)  # clipping
                    print(f"Step {global_step+1}: Grad norm: {grad_norm.item():.4f}")

                    # # Track parameter changes (if desired)
                    # param_before = next(policy_model.parameters()).clone().sum().item()
                    # print(f"Step {global_step+1}: Param sum before: {param_before:.6f}")
                    
                    # Optimizer step
                    optimizer.step()
                    
                    # # Track parameter changes after update
                    # param_after = next(policy_model.parameters()).sum().item()
                    # print(f"Step {global_step+1}: Param sum change: {param_after - param_before:.6f}")
                    
                    # Scheduler step if provided
                    if scheduler is not None:
                        scheduler.step()
                    
                    # Zero gradients after optimizer step
                    optimizer.zero_grad()
                    
                    # Update global step counter (count actual parameter updates)
                    global_step += 1
                    tokens_seen += accumulated_tokens
                    accumulated_tokens = 0

                    
                    # Evaluation step
                    if global_step % eval_freq == 0:
                        # if log_memory:
                        #     log_memory_snapshot(f"Before evaluation at step {global_step}")
                        
                        # Evaluate model
                        res = dpo_loss_fn.evaluate_dpo_loss_loader(
                            policy_model=policy_model,
                            reference_model=reference_model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            eval_iter=eval_iter
                        )
                        
                        # Track metrics
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

                        # # Check for early stopping conditions
                        # if train_reward_margin > max_reward_margin or val_reward_margin > max_reward_margin:
                        #     print(f"Training stopped: Reward margin too large ({train_reward_margin:.2f}, {val_reward_margin:.2f})")
                        #     early_stopping = True
                        #     break
                            
                        # # Stop if validation loss starts increasing
                        # if res["val_loss"] > prev_val_loss:
                        #     patience_counter += 1
                        #     if patience_counter >= patience:
                        #         print(f"Training stopped: Early stopping triggered after {patience} evaluations of increasing validation loss")
                        #         early_stopping = True
                        #         break
                        # else:
                        #     patience_counter = 0
                        #     prev_val_loss = res["val_loss"]
                
                # # Check if we need to stop early after this batch
                # if early_stopping:
                #     break
                
        finally:
            # Ensure the progress bar is closed properly
            train_loop.close()
        
        # Save batch records after each epoch to a file
        if tracking["batch_records"]:
            try:
                # Use utility function to generate standardized filename base
                records_base = get_output_filename(
                    model=config.model_name.split('/')[-1],
                    method=config.method_name.upper(),
                    file=config.training_data_filename,
                    learning_rate=config.learning_rate,
                    beta=config.beta,
                    lambda_dpop=config.lambda_dpop if hasattr(config, 'lambda_dpop') else None,
                    lambda_kl=config.lambda_kl if hasattr(config, 'lambda_kl') else None,
                    lambda_contrast=config.lambda_contrast if hasattr(config, 'lambda_contrast') else None,
                    typename="batch_records_epoch_{}.json".format(epoch+1)
                )
                records_filepath = os.path.join(config.result_dir, records_base)
                with open(records_filepath, "w") as f:
                    json.dump(tracking["batch_records"], f, indent=2)
                print(f"Saved batch records to batch_records_epoch_{epoch+1}.json")
            except Exception as e:
                print(f"Error saving batch records: {e}")

    print("Training completed.")
    return tracking