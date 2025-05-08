import os
import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from functools import partial
import copy
import time
from datetime import timedelta
import random

import src.config as config
from src.dpoLoss import DPOLoss
from src.preferenceDataset import PreferenceDataset
from src.utils import (
    get_dpo_params,
    get_output_filename,
    format_input,
    text_to_token_ids,
    generate,
    postprocess_response,
    custom_collate_fn,
    plot_losses,
    calculate_perplexity,
    log_result_csv,
    log_memory_snapshot,
    summarize_ppl_table,
    log_ppl_csv
)


# Define the training function
def train_model(
    dpo_loss_fn, optimizer, scheduler,
    policy_model, reference_model, train_loader, val_loader,
    num_epochs, eval_freq, eval_iter, gradient_accumulation_steps=1,
    log_memory=False, tokenizer=None
):
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
        "train_reward_accuracy": [],
        "val_reward_accuracy": [],
        "tokens_seen": [],
        "memory_usage": [] if log_memory else None,
        "batch_records": []  # Add a new field to store batch info
    }
    tokens_seen, global_step = 0, -1
    accumulated_tokens = 0
    prev_loss = None  # for tracking loss change
    accumulated_loss = 0.0  # for tracking loss over multiple batches

    log_csv_filename = get_output_filename(
        method=config.method_name,
        file=config.training_data_filename,
        model=config.model_name,
        learning_rate=config.learning_rate,
        beta=config.beta,
        lambda_dpop=getattr(config, "lambda_dpop", None),
        lambda_shift=getattr(config, "lambda_shift", None),
        avg=config.average_log_probs,
        typename="csv",
    )

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

                (loss, chosen_rewards, rejected_rewards, reward_accuracy,
                 policy_chosen_log_probas, policy_rejected_log_probas,
                 reference_chosen_log_probas, reference_rejected_log_probas) = dpo_loss_fn.compute_dpo_loss_batch(
                    batch=batch,
                    policy_model=policy_model,
                    reference_model=reference_model
                )
                loss = loss / gradient_accumulation_steps  # Scale loss for gradient accumulation
                accumulated_loss += loss.item() * gradient_accumulation_steps  # Accumulated unscaled loss for tracking loss over multiple batches
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
                        tracking["train_reward_accuracy"].append(res["train_reward_accuracy"])
                        tracking["val_losses"].append(res["val_loss"])
                        tracking["val_chosen_rewards"].append(res["val_chosen_reward"])
                        tracking["val_rejected_rewards"].append(res["val_rejected_reward"])
                        tracking["val_reward_accuracy"].append(res["val_reward_accuracy"])
                        if log_memory:
                            tracking["memory_usage"].append(log_memory_snapshot(f"After evaluation at step {global_step}"))
                        # Save tokens seen for this step
                        tracking["tokens_seen"].append(tokens_seen)

                        train_reward_margin = res["train_chosen_reward"] - res["train_rejected_reward"]
                        val_reward_margin = res["val_chosen_reward"] - res["val_rejected_reward"]

                        print(
                            f"Ep {epoch+1} (Step {global_step:06d}): "
                            f"Train loss {res['train_loss']:.3f}, Val loss {res['val_loss']:.3f}, "
                            f"Train reward margins {train_reward_margin:.3f}, "
                            f"Val reward margins {val_reward_margin:.3f},"
                            f" Train reward accuracy {res['train_reward_accuracy']:.3f}, "
                            f"Val reward accuracy {res['val_reward_accuracy']:.3f}, "
                            f"Policy model chosen logprobs {policy_chosen_log_probas.mean():.3f}, "
                            f"Policy model rejected logprobs {policy_rejected_log_probas.mean():.3f}"
                            f"Ref model chosen logprobs {reference_chosen_log_probas.mean():.3f}, "
                            f"Ref model rejected logprobs {reference_rejected_log_probas.mean():.3f}"
                        )

                        total_batches = len(train_loader)
                        epoch_progress = epoch + (batch_idx + 1) / total_batches

                        # Save results to CSV
                        log_result_csv(
                            filename=log_csv_filename,
                            epoch_frac=round(epoch_progress, 4),
                            step=global_step,
                            train_loss=res["train_loss"],
                            val_loss=res["val_loss"],
                            train_reward_margin=train_reward_margin,
                            val_reward_margin=val_reward_margin,
                            train_reward_accuracy=res["train_reward_accuracy"],
                            val_reward_accuracy=res["val_reward_accuracy"],
                            policy_chosen_logprobs=policy_chosen_log_probas.mean().item(),
                            policy_rejected_logprobs=policy_rejected_log_probas.mean().item(),
                            reference_chosen_logprobs=reference_chosen_log_probas.mean().item(),
                            reference_rejected_logprobs=reference_rejected_log_probas.mean().item(),
                        )

        finally:
            # Ensure the progress bar is closed properly
            train_loop.close()

        print(f"End of Epoch {epoch+1}: Total tokens seen: {tokens_seen}")

    # Save batch records after each epoch to a file
    if tracking["batch_records"]:
        try:
            # Use utils function to generate standardized filename base
            records_filepath = get_output_filename(
                method=config.method_name,
                file=config.training_data_filename,
                model=config.model_name,
                label="batch_records",
                learning_rate=config.learning_rate,
                beta=config.beta,
                lambda_dpop=config.lambda_dpop if hasattr(config, 'lambda_dpop') else None,
                lambda_shift=config.lambda_shift if hasattr(config, 'lambda_shift') else None,
                typename="json"
            )
            with open(records_filepath, "w") as f:
                json.dump(tracking["batch_records"], f, indent=2)
            print(f"Saved {len(tracking['batch_records'])} batch records to {records_filepath}")
        except Exception as e:
            print(f"Error saving batch records: {e}")

    return tracking


def run_training():
    # Get relevant parameters for the selected method
    dpo_params = get_dpo_params(config.method_name)

    # Initialize DPO loss function with only the relevant parameters
    dpo_loss_fn = DPOLoss(method=config.method_name, **dpo_params)

    # Print the parameters being used for clarity
    param_str = ", ".join([f"{k}={v}" for k, v in dpo_params.items()])
    print(f"Using {config.method_name} with {param_str}")

    # ------------------------------- Set the cache directory -------------------------------
    cache_dir = config.cache_dir  # cache directory for the Hugging Face model

    # ---------------------------- Ensure result directory exists ----------------------------
    os.makedirs(config.result_dir, exist_ok=True)

    # ----------------------- Get each filename from utils function ------------------------
    # For output text
    output_json = get_output_filename(
        method=config.method_name,
        file=config.training_data_filename,
        model=config.model_name,
        label="generated_output",
        learning_rate=config.learning_rate,
        beta=config.beta,
        lambda_dpop=config.lambda_dpop if hasattr(config, 'lambda_dpop') else None,
        lambda_shift=config.lambda_shift if hasattr(config, 'lambda_shift') else None,
        typename="json"
    )
    print("Output file path:", output_json)

    # For loss plot
    loss_plot_file = get_output_filename(
        method=config.method_name,
        file=config.training_data_filename,
        model=config.model_name,
        label="loss",
        learning_rate=config.learning_rate,
        beta=config.beta,
        lambda_dpop=config.lambda_dpop if hasattr(config, 'lambda_dpop') else None,
        lambda_shift=config.lambda_shift if hasattr(config, 'lambda_shift') else None,
        typename="png"
    )
    print("Loss plot file path:", loss_plot_file)

    # For reward margins plot
    margins_plot_file = get_output_filename(
        method=config.method_name,
        file=config.training_data_filename,
        model=config.model_name,
        label="reward_margin",
        learning_rate=config.learning_rate,
        beta=config.beta,
        lambda_dpop=config.lambda_dpop if hasattr(config, 'lambda_dpop') else None,
        lambda_shift=config.lambda_shift if hasattr(config, 'lambda_shift') else None,
        typename="png"
    )
    print("Reward margins plot file path:", margins_plot_file)

    # ---------------------------------------- Device ----------------------------------------
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # ----------------------- Load a Hugging Face model and tokenizer ------------------------
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(config.model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16)

    eos_token_id = tokenizer.eos_token_id  # Get the end of text token ID

    policy_model = model  # this is the model that will be fine-tuned
    ref_model = copy.deepcopy(model)  # create a reference model for DPO by copying and freezing the parameters
    # log_memory_snapshot("After reference model creation")

    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    print("Ref model grad status:", next(ref_model.parameters()).requires_grad)
    print("Policy model grad status:", next(policy_model.parameters()).requires_grad)

    policy_model.to(device)
    ref_model.to(device)

    # --------------------------- Set the tokenizer's padding token --------------------------
    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
    model.config.pad_token_id = tokenizer.pad_token_id  # updating model config
    tokenizer.padding_side = 'right'  # padding to right (prevent showing warning)
    print(f"Set PAD token to '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"Using EOS token '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")

    print("Model and tokenizer loaded.")

    # ------------------------------------- Load the data ------------------------------------
    try:
        with open(config.training_data_filename, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"File {config.training_data_filename} not found. Please check the path.")
        exit(1)

    print("Number of entries:", len(data))

    # ----------------------------- data set pre-processing and shuffling -----------------------------
    # Randomly select 10% of the data for testing which is fixed data in multiple runs
    random.seed(config.random_seed)  # Set seed for reproducibility
    test_size = int(len(data) * 0.1)
    test_data = random.sample(data, test_size)

    # Remove test data from the original dataset
    remaining = [d for d in data if d not in test_data]

    # Randomly shuffle the remaining data
    random.shuffle(remaining)

    # Split the remaining data into training and validation sets
    train_size = int(len(data) * 0.8)
    train_data = remaining[:train_size]
    val_data = remaining[train_size:]

    print("Train size:", train_size)
    print("Validation size:", len(remaining) - train_size)
    print("Test size:", test_size)

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))

    # ------------------------------------------------ Set warmup steps ------------------------------------------------
    # Compute the number of training steps
    batches_per_epoch = train_size // config.batch_size
    optimization_steps_per_epoch = batches_per_epoch // config.gradient_accumulation_steps
    num_training_steps = optimization_steps_per_epoch * config.num_epochs

    # Dynamic warmup steps
    num_warmup_steps = int(0.1 * num_training_steps)
    print(f"Dataset size: {len(data)}, num_training_steps: {num_training_steps}, num_warmup_steps: {num_warmup_steps}")
    # ------------------------------------------------------------------------------------------------------------------

    # ---------------------------- Custom collate function for DataLoader ---------------------------
    customized_collate_fn = partial(
        custom_collate_fn,
        tokenizer=tokenizer,
        device=device,
        allowed_max_length=config.allowed_max_length,  # The supported context length of the model
        mask_prompt=True
    )

    # ---------------------------- Create datasets and dataloaders ---------------------------
    train_dataset = PreferenceDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=customized_collate_fn,
        drop_last=True,
        shuffle=True
    )

    val_dataset = PreferenceDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=customized_collate_fn,
        drop_last=False,
        shuffle=False
    )

    test_dataset = PreferenceDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
    )

    validation_ppl_loader = DataLoader(
        val_dataset,
        batch_size=config.val_ppl_batch_size,
        collate_fn=customized_collate_fn,
        drop_last=False,
        shuffle=False
    )

    # print("Train loader:")
    # for batch in train_loader:
    #     print(
    #         batch["chosen"].shape,
    #         batch["rejected"].shape,
    #     )
    # print("\n")

    # log_memory_snapshot("Before initial evaluation")

    # Evaluate initial state
    # print("\nEvaluating initial state...")
    # res = dpo_loss_fn.evaluate_dpo_loss_loader(
    #     policy_model=model,
    #     reference_model=ref_model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     eval_iter=5
    # )
    # print("Training loss:", res["train_loss"])
    # print("Validation loss:", res["val_loss"])

    # print("Train reward margin:", res["train_chosen_reward"] - res["train_rejected_reward"])
    # print("Val reward margin:", res["val_chosen_reward"] - res["val_rejected_reward"])

    # log_memory_snapshot("After initial evaluation")

    # ---- Log the reference model's perplexity on the chosen samples, rejected samples, self-generated samples ----
    ppl_filename = get_output_filename(
        method=config.method_name,
        file=config.training_data_filename,
        model=config.model_name,
        label="ppl",
        learning_rate=config.learning_rate,
        beta=config.beta,
        lambda_dpop=getattr(config, "lambda_dpop", None),
        lambda_shift=getattr(config, "lambda_shift", None),
        typename="csv"
    )

    print(f"\nPerplexity log file path: {ppl_filename}")
    print("Computing reference model perplexity...\n")
    ppl_start_time = time.time()

    ref_ppl = summarize_ppl_table(ref_model, tokenizer, validation_ppl_loader, device)
    log_ppl_csv(
        filename=ppl_filename,
        model_name="reference",
        chosen_ppl=ref_ppl["chosen"],
        rejected_ppl=ref_ppl["rejected"],
        self_ppl=ref_ppl["self"]
    )

    ppl_end_time = time.time()
    ppl_execution_time = (ppl_end_time - ppl_start_time) / 60
    print(f"Reference model PPL computed and saved in {ppl_execution_time:.2f} minutes (in {str(timedelta(seconds=ppl_end_time - ppl_start_time))})")

    print("\n[REF] PPL")
    print(f"chosen:{ref_ppl['chosen']:.2f}  rejected:{ref_ppl['rejected']:.2f}  self‑gen:{ref_ppl['self']:.2f}\n")

    # Before starting the training, print the initail losses and rewards:
    print("=" * 50)
    print("\nStarting training...")
    print("=" * 50)
    # log_memory_snapshot("Before training")

    # Initialize the optimizer and scheduler
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    start_time = time.time()

    torch.manual_seed(123)  # For reproducibility due to the shuffling in the data loader

    tracking = train_model(
        dpo_loss_fn=dpo_loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        policy_model=policy_model,
        reference_model=ref_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.num_epochs,
        eval_freq=config.eval_freq,
        eval_iter=5,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        tokenizer=tokenizer,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes (in {str(timedelta(seconds=end_time - start_time))})")
    print(f" with {config.method_name}, {config.training_data_filename}, {config.model_name}, "
          f"lr={config.learning_rate}, beta={config.beta}, lambda_dpop={config.lambda_dpop}, lambda_shift={config.lambda_shift}")

    # log_memory_snapshot("After training")

    train_margin = tracking['train_chosen_rewards'][-1] - tracking['train_rejected_rewards'][-1]
    val_margin = tracking['val_chosen_rewards'][-1] - tracking['val_rejected_rewards'][-1]
    train_acc = tracking['train_reward_accuracy'][-1]
    val_acc = tracking['val_reward_accuracy'][-1]
    print("Final train/validation statistics:")
    print(f"Train loss: {tracking['train_losses'][-1]}")
    print(f"Validation loss: {tracking['val_losses'][-1]}")
    print(f"Train reward margin: {train_margin:.3f}")
    print(f"Validation reward margin: {val_margin:.3f}")
    print(f"Train reward accuracy: {train_acc:.3f}")
    print(f"Validation reward accuracy: {val_acc:.3f}")
    print(f"Tokens seen: {tracking['tokens_seen'][-1]}")

    # print("\nAnalyzing batch records for significant loss changes:")
    # if "batch_records" in tracking and tracking["batch_records"]:
    #     # Find batches with the largest loss increases and decreases
    #     sorted_records = sorted(tracking["batch_records"], key=lambda x: x["loss_change"])

    #     # Top 3 decreases (improvements)
    #     print("\nTop 3 Loss Decreases (Improvements):")
    #     for record in sorted_records[:3]:
    #         print(f"Batch {record['batch_idx']} - Loss change: {record['loss_change']:.4f}")
    #         print(f"Reward difference: {record['reward_diff']:.4f}")

    #     # Top 3 increases (deteriorations)
    #     print("\nTop 3 Loss Increases (Deteriorations):")
    #     for record in sorted_records[-3:]:
    #         print(f"Batch {record['batch_idx']} - Loss change: {record['loss_change']:.4f}")
    #         print(f"Reward difference: {record['reward_diff']:.4f}")
    # else:
    #     print("No batch records found in tracking data")

    # Save the model and tokenizer
    save_path = config.fine_tuned_model_path
    policy_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")

    print("\nComputing policy model perplexity...\n")
    ppl_start_time = time.time()
    pol_ppl = summarize_ppl_table(policy_model, tokenizer, validation_ppl_loader, device)
    log_ppl_csv(
        filename=ppl_filename,
        model_name="policy",
        chosen_ppl=pol_ppl["chosen"],
        rejected_ppl=pol_ppl["rejected"],
        self_ppl=pol_ppl["self"]
    )
    ppl_end_time = time.time()
    ppl_execution_time = (ppl_end_time - ppl_start_time) / 60
    print(f"Policy model PPL computed and saved in {ppl_execution_time:.2f} minutes (in {str(timedelta(seconds=ppl_end_time - ppl_start_time))})")
    print("\n[POL] PPL")
    print(f"chosen:{pol_ppl['chosen']:.2f}  rejected:{pol_ppl['rejected']:.2f}  self‑gen:{pol_ppl['self']:.2f}\n")

    # Plot the losses
    epochs_tensor = torch.linspace(0, config.num_epochs, len(tracking["train_losses"]))
    plot_losses(
        epochs_seen=epochs_tensor,
        tokens_seen=tracking["tokens_seen"],
        train_losses=tracking["train_losses"],
        val_losses=tracking["val_losses"],
        save_path=loss_plot_file,
        label="loss"
    )

    train_reward_margins = [i - j for i, j in zip(tracking["train_chosen_rewards"], tracking["train_rejected_rewards"])]
    val_reward_margins = [i - j for i, j in zip(tracking["val_chosen_rewards"], tracking["val_rejected_rewards"])]

    plot_losses(
        epochs_seen=epochs_tensor,
        tokens_seen=tracking["tokens_seen"],
        train_losses=train_reward_margins,
        val_losses=val_reward_margins,
        save_path=margins_plot_file,
        label="reward margin"
    )

    fine_tuned_tokenizer = AutoTokenizer.from_pretrained(save_path)
    fine_tuned_model = AutoModelForCausalLM.from_pretrained(save_path)
    print("Tuned model's tokenizer loaded.")

    ref_model.to(device)  # Ensure reference model is on device
    fine_tuned_model.to(device)  # Ensure fine-tuned model is on device

    # Use the test data to evaluate the fine-tuned model
    print("Starting test evaluation...")
    test_res = dpo_loss_fn.evaluate_dpo_loss_loader(
        policy_model=fine_tuned_model,
        reference_model=ref_model,
        train_loader=None,
        val_loader=test_loader,
        eval_iter=5
    )

    print("Test loss:", test_res["val_loss"])
    print("Test reward margin:", test_res["val_chosen_reward"] - test_res["val_rejected_reward"])

    # List for storing the test results to write to the json file
    test_results = []

    # Check first entry to determine data type
    input_key = "question" if "question" in test_data[0] else "instruction"

    # Check the maximum sequence length in the test data
    # max_length = 0
    # for entry in test_data:
    #     tokens = fine_tuned_tokenizer(format_input(entry), add_special_tokens=True).input_ids
    #     max_length = max(max_length, len(tokens))
    # print(f"Test data max sequence length: {max_length}")

    # # Set stride based on the maximum length
    # if max_length > config.allowed_max_length:
    #     print("Warning: Long sequences detected, using stride=512")
    #     stride = 512
    # else:
    #     stride = None

    if config.EVAL_USE_SAMPLING:
        print("Using sampling for evaluation")
        eval_temperature = config.temperature
        eval_top_p = config.top_p
    else:
        print("Using greedy decoding for evaluation")
        eval_temperature = 0.0
        eval_top_p = None

    try:
        test_results = []
        test_start_time = time.time()
        for batch in tqdm(test_loader, desc="Test Eval"):
            questions = batch["question_texts"]
            expected_pure_resps = batch["chosen_texts"]  # the chosen response without the prompt
            
            # batch["prompt"] is a tensor of token ids (bsz, seq_len)
            # 1) decoding prompt
            full_prompts = [tokenizer.decode(ids, skip_special_tokens=False) for ids in batch["prompt"]]
            
            # 2) Batch generation of responses using the reference and fine-tuned models 
            input_ids = batch["prompt"].to(device)
            with torch.no_grad():
                ref_out = ref_model.generate(input_ids, max_new_tokens=config.max_new_tokens, do_sample=False)
                pol_out = fine_tuned_model.generate(input_ids, max_new_tokens=config.max_new_tokens, do_sample=False)
            
            # 3) Batch decoding the generated responses
            ref_resps = [postprocess_response(tokenizer.decode(o, skip_special_tokens=True)) 
                        for o in ref_out]
            pol_resps = [postprocess_response(tokenizer.decode(o, skip_special_tokens=True)) 
                        for o in pol_out]

            ref_ful_resps = [f"{p}{r}{tokenizer.eos_token}" for p, r in zip(full_prompts, ref_resps)]
            pol_ful_resps = [f"{p}{r}{tokenizer.eos_token}" for p, r in zip(full_prompts, pol_resps)]
            exp_full_resps = [f"{p}{r}{tokenizer.eos_token}" for p, r in zip(full_prompts, expected_pure_resps)]
            
            # 4) Calculate perplexity for the reference and fine-tuned models and store them 
            ref_ppls = calculate_perplexity(ref_model, tokenizer, ref_ful_resps, max_length=config.allowed_max_length, device=device)
            pol_ppls = calculate_perplexity(fine_tuned_model, tokenizer, pol_ful_resps, max_length=config.allowed_max_length, device=device)
            ref_exp_ppls = calculate_perplexity(ref_model, tokenizer, exp_full_resps, max_length=config.allowed_max_length, device=device)
            pol_exp_ppls = calculate_perplexity(fine_tuned_model, tokenizer, exp_full_resps, max_length=config.allowed_max_length, device=device)
            
            # 5) Print the results and store them in the dictionary
            for i, question in enumerate(questions):
                # Use the previously determined input key
                print(f"\nInput {i}:\n {question}")

                print("\n ----- Reference Model ----- ")
                print(f"Reference Response: {ref_resps[i]}")
                print(f"Perplexity: {ref_ppls[i]:.2f}")

                print("\n ----- Policy Model ----- ")
                print(f"Policy Response: {pol_resps[i]}")
                print(f"Perplexity: {pol_ppls[i]:.2f}")

                print("\n ----- Expected Response ----- ")
                print(f"Expected Answer: {expected_pure_resps[i]}")
                print(f"Gold Answer PPL (ref):    {ref_exp_ppls[i]:.2f}")
                print(f"Gold Answer PPL (policy): {pol_exp_ppls[i]:.2f}")
                print("=" * 80, "\n")

                test_results.append({
                    "input": question,
                    "ref_response": ref_resps[i],
                    "policy_response": pol_resps[i],
                    "expected": expected_pure_resps[i],
                    "ref_perplexity": ref_ppls[i] if isinstance(ref_ppls, list) else ref_ppls,
                    "policy_perplexity": pol_ppls[i] if isinstance(pol_ppls, list) else pol_ppls, 
                    "ref_gold_answer_perplexity": ref_exp_ppls[i] if isinstance(ref_exp_ppls, list) else ref_exp_ppls,
                    "policy_gold_answer_perplexity": pol_exp_ppls[i] if isinstance(pol_exp_ppls, list) else pol_exp_ppls
                })

        test_end_time = time.time()
        test_execution_time = (test_end_time - test_start_time) / 60
        print(f"Test evaluation completed in {test_execution_time:.2f} minutes (in {str(timedelta(seconds=test_end_time - test_start_time))})")

    except KeyboardInterrupt:
        print("\nInterrupted! Saving partial results...")

    finally:
        # Save the test results to a JSON file
        with open(output_json, "w") as f:
            json.dump(test_results, f, indent=4)
        print("Test results saved to:", output_json)
