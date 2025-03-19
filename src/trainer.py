import torch
import src.config as config
from tqdm import tqdm

# Define the training function
def train_model_dpo_simple(
    dpo_loss_fn, optimizer,
    policy_model, reference_model, train_loader, val_loader,
    num_epochs, eval_freq, eval_iter):
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
    # sample_entry = val_data[0] if val_data else None # Sample entry for generation

    # Main training loop
    for epoch in range(num_epochs):
        policy_model.train()  # Set model to training mode

        # Add tqdm progress bar for each epoch
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for batch_idx, batch in enumerate(train_loop):

            optimizer.zero_grad()
            # with autocast():  # Enable mixed precision
            loss, chosen_rewards, rejected_rewards = dpo_loss_fn.compute_dpo_loss_batch(
                    batch=batch,
                    policy_model=policy_model,
                    reference_model=reference_model
            )
            loss.backward()  # Compute gradients

            # Gradient clipping, monitor if learning rate is appropriate or if the model is experiencing gradient issues.
            grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)  # clipping
            print(f"Step {global_step+1}: Grad norm: {grad_norm.item():.4f}")

            # param_before = next(policy_model.parameters()).clone().sum().item()
            # print(f"Step {global_step+1}: Param sum before: {param_before:.6f}")

            optimizer.step() # Direct optimizer step
            # scheduler.step() # Update learning rate

            # param_after = next(policy_model.parameters()).sum().item()
            # print(f"Step {global_step+1}: Param sum change: {param_after - param_before:.6f}")

            # Track tokens processed
            tokens_seen += batch["chosen"].numel()
            global_step += 1

            # evaluation step
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