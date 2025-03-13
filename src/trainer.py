import torch

# Define the training function
def train_model_dpo_simple(
    dpo_loss_fn, optimizer, scheduler,
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

    # sample_entry = val_data[0] if val_data else None # Sample entry for generation

    # Main training loop
    for epoch in range(num_epochs):
        policy_model.train()  # Set model to training mode

        for batch_idx, batch in enumerate(train_loader):

            optimizer.zero_grad()
            # with autocast():  # Enable mixed precision
            loss, chosen_rewards, rejected_rewards = dpo_loss_fn.compute_dpo_loss_batch(
                    batch=batch,
                    policy_model=policy_model,
                    reference_model=reference_model
            )
            # print(f"Step {global_step+1}: Loss before backward: {loss.item():.4f}")
            loss.backward()  # Direct backward pass without scaling
            grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)  # Optional clipping
            # print(f"Step {global_step+1}: Grad norm: {grad_norm.item():.4f}")
            param_before = next(policy_model.parameters()).clone().sum().item()
            optimizer.step() # Direct optimizer step
            param_after = next(policy_model.parameters()).sum().item()
            # print(f"Step {global_step+1}: Param sum change: {param_after - param_before:.6f}")
            scheduler.step()  # Update learning rate after optimizer step

            # tokens_seen = torch.tensor(0, dtype=torch.int64) # avoid overflow by using torch.tensor with dtype int64
            tokens_seen += batch["chosen"].numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                res = dpo_loss_fn.evaluate_dpo_loss_loader(
                    policy_model=policy_model,
                    reference_model=reference_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    eval_iter=eval_iter
                )
                # if sample_entry and (global_step // eval_freq) % 2 == 0:  # generate every 2nd evaluation
                #     policy_model.eval()
                #     with torch.no_grad():
                #         try:
                #             # prepare input
                #             input_text = format_input(sample_entry)
                #             token_ids = text_to_token_ids(input_text, tokenizer).to(device)
                            
                #             # generation config
                #             generation_config = {
                #                 'max_new_tokens': max_new_tokens,
                #                 'temperature': temperature,
                #                 'top_p': top_p,
                #                 'eot_token_id': eot_token_id
                #             }
                            
                #             # execute generation
                #             generated = generate(
                #                 model=policy_model,
                #                 idx=token_ids, #.to(device),
                #                 stopping_criteria=stopping_criteria,
                #                 **generation_config
                #             )
                            
                #             # post-process the generated text
                #             full_text = token_ids_to_text(generated, tokenizer)
                #             response = full_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                #             response = response.split("<|eot_id|>")[0].strip()

                #         except Exception as e:
                #             response = f"~~~ Generation Error: {str(e)}"
                #             print(f"Generation failed at step {global_step}: {str(e)}")

                #         finally:
                #             policy_model.train()

                #     # Print the generated response
                #     print(f"\n{'='*40} Generation Sample (Step {global_step}) {'='*40}")
                #     print(f"[Input]\n{sample_entry['question']}")
                #     print(f"\n[Generated Response]\n{response}")
                #     print(f"[Expected Response]\n{sample_entry['chosen']}")
                #     print('='*90 + '\n')

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
    print("Training completed.")
    return tracking