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
    prev_loss = None  # for tracking loss change
    accumulated_loss = 0.0 # for tracking loss over multiple batches

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
                accumulated_loss += loss.item() * gradient_accumulation_steps # Accumulated unscaled loss for tracking loss over multiple batches
                loss.backward()  # Compute gradients

                # Calculate the number of tokens in the batch, accumulate tokens seen for the current batch
                batch_token_count = batch["chosen"].numel() + batch["rejected"].numel()
                accumulated_tokens += batch_token_count

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

                    # Print tokens seen every 10 steps
                    if global_step % 10 == 0:
                        print(f"Step {global_step}: Tokens seen: {tokens_seen}")

                    # Calculate average loss over the accumulated steps
                    # Note: gradient_accumulation_steps is used to determine how many batches were accumulated before the optimizer step
                    steps_in_accumulation = min(gradient_accumulation_steps, 
                                                batch_idx + 1 - (batch_idx // gradient_accumulation_steps) * gradient_accumulation_steps)
                    avg_loss = accumulated_loss / steps_in_accumulation
                    loss_change = avg_loss - prev_loss if prev_loss is not None else 0.0

                    # loss detection
                    if prev_loss is not None:
                        if abs(loss_change) > 0.1 and tokenizer is not None:
                            batch_record = {
                                "epoch": epoch + 1,
                                "step": global_step,
                                "batch_idx": batch_idx,
                                "loss": avg_loss,
                                "loss_change": loss_change,
                                "reward_diff": chosen_rewards.item() - rejected_rewards.item()
                            }
                            
                            sample_idx = 0
                            try:
                                chosen_text = postprocess_response(tokenizer.decode(batch["chosen"][sample_idx]))
                                rejected_text = postprocess_response(tokenizer.decode(batch["rejected"][sample_idx]))
                                if len(chosen_text) > 200:
                                    chosen_text = chosen_text[:200] + "..."
                                if len(rejected_text) > 200:
                                    rejected_text = rejected_text[:200] + "..."
                                batch_record["sample_chosen"] = chosen_text
                                batch_record["sample_rejected"] = rejected_text
                                print(f"\n{'=='*40}\nLoss Change: {loss_change:.4f} at step {global_step}")
                                print(f"New loss: {avg_loss:.4f}, Previous loss: {prev_loss:.4f}")
                                if scheduler is not None:
                                    current_lr = scheduler.get_last_lr()[0]
                                    print(f"Current learning rate: {current_lr:.6f}")
                                print(f"Chosen sample: {chosen_text[:100]}...")
                                print(f"Rejected sample: {rejected_text[:100]}...")
                                print(f"{'=='*40}\n")

                            except Exception as e:
                                print(f"Error decoding batch: {e}")
                                batch_record["sample_chosen"] = "Decode failed"
                                batch_record["sample_rejected"] = "Decode failed"
                            tracking["batch_records"].append(batch_record)

                    prev_loss = avg_loss
                    accumulated_loss = 0.0  # Reset tracking accumulated loss for next gradient accumulation step

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
                
        finally:
            # Ensure the progress bar is closed properly
            train_loop.close()

        print(f"End of Epoch {epoch+1}: Total tokens seen: {tokens_seen}")
        
    # Save batch records after each epoch to a file
    if tracking["batch_records"]:
        try:
            # Use utility function to generate standardized filename base
            records_base = get_output_filename(
                model=config.model_name.split('/')[-1],
                method=config.method_name.upper(),
                file=config.training_data_filename,
                label="batch_records",
                learning_rate=config.learning_rate,
                beta=config.beta,
                lambda_dpop=config.lambda_dpop if hasattr(config, 'lambda_dpop') else None,
                lambda_shift=config.lambda_shift if hasattr(config, 'lambda_shift') else None,
                typename="json"
            )
            records_filepath = os.path.join(config.result_dir, records_base)
            with open(records_filepath, "w") as f:
                json.dump(tracking["batch_records"], f, indent=2)
            print(f"Saved {len(tracking['batch_records'])} batch records to {records_filepath}")
        except Exception as e:
            print(f"Error saving batch records: {e}")    

    return tracking