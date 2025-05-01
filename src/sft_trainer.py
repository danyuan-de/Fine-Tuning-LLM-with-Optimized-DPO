import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from preferenceDataset import PreferenceDataset
import src.config as config


def run_sft(train_data):
    """
    Supervised Fine-Tuning (SFT) on chosen completions.
    Trains the model with cross-entropy on prompt + chosen text pairs.
    Saves the fine-tuned model to config.sft_output_dir.
    """
    # ---------------------------------------- Device ----------------------------------------
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Prepare tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset with only chosen completions
    train_dataset = PreferenceDataset(
        data_path=train_data,
        tokenizer=tokenizer,
        split="train",
        only_chosen=True  # custom flag to filter only chosen
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.sft_batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    device = torch.device()
    model.to(device)
    model.train()

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.sft_lr,
        weight_decay=config.sft_weight_decay
    )
    total_steps = len(train_loader) * config.sft_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.sft_warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    for epoch in range(config.sft_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # labels are the same as input_ids for next-token prediction
            labels = input_ids.clone()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            if (step + 1) % config.log_interval == 0:
                avg = epoch_loss / (step + 1)
                print(f"[SFT] Epoch {epoch+1}/{config.sft_epochs}, Step {step+1}/{len(train_loader)}, Loss: {avg:.4f}")

    # Save fine-tuned model
    os.makedirs(config.sft_output_dir, exist_ok=True)
    model.save_pretrained(config.sft_output_dir)
    tokenizer.save_pretrained(config.sft_output_dir)
    print(f"Supervised fine-tuned model saved to {config.sft_output_dir}")
