#!/usr/bin/env python
# Description: This script trains a model using DPO on multiple GPUs
# Execute: python -m torch.distributed.launch --nproc_per_node=NUM_GPUS src/main_distributed.py

import os
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList
from functools import partial
import copy
import time
from datetime import timedelta
import numpy as np
import argparse
import sys

import src.config as config
from src.dpoLoss import DPOLoss
from src.preferenceDataset import PreferenceDataset
from src.utility import *
from src.trainer import train_model_dpo_distributed
from src.distributed_utils import setup_distributed, cleanup_distributed, is_main_process, wrap_model_distributed

def main(rank, world_size, args):
    """
    Main function for distributed training
    
    Args:
        rank (int): The rank of the current process
        world_size (int): Total number of processes
        args (argparse.Namespace): Command line arguments
    """
    # Initialize distributed environment
    setup_distributed(rank, world_size)
    
    # --------- File Paths ---------
    model_workspace_dir = config.model_workspace_dir
    cache_dir = config.cache_dir
    result_dir = config.result_dir
    model_name = config.model_name
    file_path = config.file_content
    
    # Create result directory if it doesn't exist
    if is_main_process(rank):
        os.makedirs(result_dir, exist_ok=True)

    # --------- Hyperparameters ---------
    allowed_max_length = config.allowed_max_length
    max_new_tokens = config.max_new_tokens
    batch_size = config.batch_size // world_size  # Adjust batch size per GPU
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate * world_size  # Scale learning rate with world size
    weight_decay = config.weight_decay
    temperature = config.temperature
    top_p = config.top_p
    dpo_loss_fn = DPOLoss(beta=config.beta, lambda_kl=config.lambda_kl)
    
    # --- End of Text Token ---
    eot_token = config.eot_token
    
    # ----- Set device for this process -----
    device = rank  # In distributed training, device = rank
    
    # Only print on main process to avoid clutter
    if is_main_process(rank):
        print(f"Using {world_size} GPUs for training")
        print(f"Process {rank} using device: cuda:{rank}")
    
    # ----- Load tokenizer and model -----
    # Load tokenizer (this is not distributed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    # Add special tokens to tokenizer
    special_tokens = {
        "additional_special_tokens": [eot_token]
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Ensure pad_token is defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if is_main_process(rank):
            print(f"Using EOS token '{tokenizer.eos_token}' as PAD token")
    
    # Add the EOT token to the tokenizer
    eot_token_id = tokenizer.convert_tokens_to_ids(eot_token)
    
    # ----- Load models -----
    # Load model with half precision to save memory
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        cache_dir=cache_dir, 
        torch_dtype=torch.bfloat16,
        device_map={"": rank}  # Assign to specific GPU
    )
    model.resize_token_embeddings(len(tokenizer))  # adjust for new tokens
    
    # Create policy and reference models
    policy_model = model  # this is the model that will be fine-tuned
    ref_model = copy.deepcopy(model)  # create a reference model for DPO
    
    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    
    if is_main_process(rank):
        print("Models loaded successfully")
        print("Ref model grad status:", next(ref_model.parameters()).requires_grad)
        print("Policy model grad status:", next(policy_model.parameters()).requires_grad)
    
    # ----- Wrap models in DistributedDataParallel -----
    policy_model = wrap_model_distributed(policy_model, rank)
    # Note: Reference model doesn't need DDP since it's frozen, but we keep it consistent
    ref_model = wrap_model_distributed(ref_model, rank)
    
    # ----- Load and prepare data -----
    # Only the main process loads the data
    if is_main_process(rank):
        print(f"Loading data from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        print(f"Number of entries: {len(data)}")
        
        # Train/val/test split
        train_portion = int(len(data) * 0.85)
        test_portion = int(len(data) * 0.1)
        val_portion = len(data) - train_portion - test_portion
        
        print(f"Train portion: {train_portion}")
        print(f"Validation portion: {val_portion}")
        print(f"Test portion: {test_portion}")
        
        train_data = data[:train_portion]
        test_data = data[train_portion:train_portion + test_portion]
        val_data = data[train_portion + test_portion:]
        
        print(f"Training set length: {len(train_data)}")
        print(f"Validation set length: {len(val_data)}")
        print(f"Test set length: {len(test_data)}")
    else:
        train_data, test_data, val_data = None, None, None
    
    # Synchronize all processes to ensure data is loaded
    dist.barrier()
    
    # Broadcast data from rank 0 to all processes
    if not is_main_process(rank):
        train_data = [None]
        val_data = [None]
        test_data = [None]
    
    # Broadcast train_data
    train_data_obj = [train_data] if is_main_process(rank) else [None]
    dist.broadcast_object_list(train_data_obj, src=0)
    if not is_main_process(rank):
        train_data = train_data_obj[0]
    
    # Broadcast val_data
    val_data_obj = [val_data] if is_main_process(rank) else [None]
    dist.broadcast_object_list(val_data_obj, src=0)
    if not is_main_process(rank):
        val_data = val_data_obj[0]
    
    # Broadcast test_data
    test_data_obj = [test_data] if is_main_process(rank) else [None]
    dist.broadcast_object_list(test_data_obj, src=0)
    if not is_main_process(rank):
        test_data = test_data_obj[0]
    
    # Create custom collate function
    customized_collate_fn = partial(
        custom_collate_fn,
        eot_token_id=eot_token_id,
        tokenizer=tokenizer,
        device=device,
        mask_prompt_tokens=True,
        allowed_max_length=allowed_max_length
    )
    
    # Create datasets
    train_dataset = PreferenceDataset(train_data, tokenizer)
    val_dataset = PreferenceDataset(val_data, tokenizer)
    test_dataset = PreferenceDataset(test_data, tokenizer)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create dataloaders with distributed samplers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        sampler=val_sampler,
        drop_last=False,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        sampler=test_sampler,
        drop_last=False,
        pin_memory=True
    )
    
    if is_main_process(rank):
        print("DataLoaders created successfully")
    
    # Set up stopping criteria for generation
    stopping_criteria = StoppingCriteriaList([
        EOTStoppingCriteria(eot_token_id=eot_token_id)
    ])
    
    # Create optimizer - each process has its own optimizer for its model shard
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create scheduler if needed
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * len(train_loader),
        eta_min=1e-6
    )
    
    # Only show samples in rank 0 process
    if is_main_process(rank):
        batch = next(iter(train_loader))
        print("Chosen sample:", tokenizer.decode(batch["chosen"][0].tolist()))
        print("Rejected sample:", tokenizer.decode(batch["rejected"][0].tolist()))
        
        # Evaluate initial model
        res = dpo_loss_fn.evaluate_dpo_loss_loader(
            policy_model=policy_model,
            reference_model=ref_model,
            train_loader=train_loader,
            val_loader=val_loader,
            eval_iter=5
        )
        
        print("Initial training loss:", res["train_loss"])
        print("Initial validation loss:", res["val_loss"])
        print("Initial train reward margin:", res["train_chosen_reward"] - res["train_rejected_reward"])
        print("Initial val reward margin:", res["val_chosen_reward"] - res["val_rejected_reward"])
    
    # Make sure all processes are in sync before starting training
    dist.barrier()
    
    # Start training timer
    start_time = time.time()
    
    # Set deterministic training for reproducibility
    if args.deterministic:
        torch.manual_seed(rank + 42)  # Different seed for each process
        torch.cuda.manual_seed(rank + 42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Train the model
    tracking = train_model_dpo_distributed(
        dpo_loss_fn=dpo_loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        policy_model=policy_model,
        reference_model=ref_model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        rank=rank,
        world_size=world_size
    )
    
    # End training timer
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    
    # Save results and evaluate only on the main process
    if is_main_process(rank):
        print(f"Training completed in {execution_time_minutes:.2f} minutes")
        
        # Save the model and tokenizer
        save_path = config.fine_tuned_model_path
        policy_model.module.save_pretrained(save_path)  # Use .module to access the underlying model
        tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to {save_path}")
        
        # Plot the losses
        epochs_tensor = torch.linspace(0, num_epochs, len(tracking["train_losses"]))
        plot_losses(
            epochs_seen=epochs_tensor,
            tokens_seen=tracking["tokens_seen"],
            train_losses=tracking["train_losses"],
            val_losses=tracking["val_losses"],
            label="DPO_loss"
        )
        
        # Plot reward margins
        train_reward_margins = [i-j for i, j in zip(tracking["train_chosen_rewards"], tracking["train_rejected_rewards"])]
        val_reward_margins = [i-j for i, j in zip(tracking["val_chosen_rewards"], tracking["val_rejected_rewards"])]
        
        plot_losses(
            epochs_seen=epochs_tensor,
            tokens_seen=tracking["tokens_seen"],
            train_losses=train_reward_margins,
            val_losses=val_reward_margins,
            label="reward_margins"
        )
        
        # Load the saved model for evaluation
        fine_tuned_tokenizer = AutoTokenizer.from_pretrained(save_path)
        fine_tuned_model = AutoModelForCausalLM.from_pretrained(save_path)
        fine_tuned_model.to(device)
        
        print("Starting evaluation...")
        # Evaluate the fine-tuned model on the validation set
        val_res = dpo_loss_fn.evaluate_dpo_loss_loader(
            policy_model=fine_tuned_model,
            reference_model=ref_model.module,  # Use .module to access the underlying model
            train_loader=None,
            val_loader=val_loader,
            eval_iter=5
        )
        
        print("Evaluation loss:", val_res["val_loss"])
        print("Evaluation reward margin:", val_res["val_chosen_reward"] - val_res["val_rejected_reward"])
        
        # Generate examples from validation set
        for i, entry in enumerate(val_data[:3]):
            input_text = format_input(entry)
            
            # Reference Model Generation
            ref_input_ids = text_to_token_ids(input_text, tokenizer).to(device)
            ref_generated = generate(
                model=ref_model.module,  # Use .module to access the underlying model
                idx=ref_input_ids,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
                eot_token_id=eot_token_id
            )
            ref_full_text = tokenizer.decode(ref_generated[0], skip_special_tokens=False)
            ref_response = postprocess_response(ref_full_text)
            
            # Fine-Tuned Model Generation
            fine_tuned_input_ids = text_to_token_ids(input_text, fine_tuned_tokenizer).to(device)
            fine_tuned_generated = generate(
                model=fine_tuned_model,
                idx=fine_tuned_input_ids,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
                eot_token_id=eot_token_id
            )
            fine_tuned_full_text = fine_tuned_tokenizer.decode(fine_tuned_generated[0], skip_special_tokens=False)
            fine_tuned_response = postprocess_response(fine_tuned_full_text)
            
            print(f"\nInput{i}: {entry['question']}")
            print("\n ----- Reference Model ----- ")
            print(f"Reference Response: {ref_response}")
            print("\n ----- Fine-Tuned Model ----- ")
            print(f"Fine-Tuned Response: {fine_tuned_response}")
            print("\n ----- Expected Response ----- ")
            print(f"Expected Answer: {entry['chosen']}")
            print("="*80)
            
            # Save to output file
            with open(os.path.join(result_dir, "distributed_output_val.txt"), "a") as f:
                f.write(f"\nInput{i}: {entry['question']}\n")
                f.write("\n ----- Reference Model ----- \n")
                f.write(f"Reference Response: {ref_response}\n")
                f.write("\n ----- Fine-Tuned Model ----- \n")
                f.write(f"Fine-Tuned Response: {fine_tuned_response}\n")
                f.write("\n ----- Expected Response ----- \n")
                f.write(f"Expected Answer: {entry['chosen']}\n")
                f.write("="*80 + "\n")
        
        # Test evaluation
        print("Starting test evaluation...")
        test_res = dpo_loss_fn.evaluate_dpo_loss_loader(
            policy_model=fine_tuned_model,
            reference_model=ref_model.module,
            train_loader=None,
            val_loader=test_loader,
            eval_iter=5
        )
        
        print("Test loss:", test_res["val_loss"])
        print("Test reward margin:", test_res["val_chosen_reward"] - test_res["val_rejected_reward"])
        
        # Generate examples from test set
        for i, entry in enumerate(test_data[:5]):
            input_text = format_input(entry)
            
            # Reference Model Generation
            ref_input_ids = text_to_token_ids(input_text, tokenizer).to(device)
            ref_generated = generate(
                model=ref_model.module,
                idx=ref_input_ids,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
                eot_token_id=eot_token_id
            )
            ref_full_text = tokenizer.decode(ref_generated[0], skip_special_tokens=False)
            ref_response = postprocess_response(ref_full_text)
            
            # Fine-Tuned Model Generation
            fine_tuned_input_ids = text_to_token_ids(input_text, fine_tuned_tokenizer).to(device)
            fine_tuned_generated = generate(
                model=fine_tuned_model,
                idx=fine_tuned_input_ids,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
                eot_token_id=eot_token_id
            )
            fine_tuned_full_text = fine_tuned_tokenizer.decode(fine_tuned_generated[0], skip_special_tokens=False)
            fine_tuned_response = postprocess_response(fine_tuned_full_text)
            
            print(f"\nInput{i}: {entry['question']}")
            print("\n ----- Reference Model ----- ")
            print(f"Reference Response: {ref_response}")
            print("\n ----- Fine-Tuned Model ----- ")
            print(f"Fine-Tuned Response: {fine_tuned_response}")
            print("\n ----- Expected Response ----- ")
            print(f"Expected Answer: {entry['chosen']}")
            print("="*80)
            
            # Save to output file
            with open(os.path.join(result_dir, "distributed_output_test.txt"), "a") as f:
                f.write(f"\nInput{i}: {entry['question']}\n")
                f.write("\n ----- Reference Model ----- \n")
                f.write(f"Reference Response: {ref_response}\n")
                f.write("\n ----- Fine-Tuned Model ----- \n")
                f.write(f"Fine-Tuned Response: {fine_tuned_response}\n")
                f.write("\n ----- Expected Response ----- \n")
                f.write(f"Expected Answer: {entry['chosen']}\n")
                f.write("="*80 + "\n")
    
    # Clean up distributed environment
    cleanup_distributed()


if __name__ == "__main__":
    # Fix for torch.distributed.launch using --local-rank and argparse expecting --local_rank
    # This replaces all occurrences of "--local-rank" with "--local_rank" in the arguments
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--local-rank'):
            sys.argv[i] = arg.replace('--local-rank', '--local_rank')
    
    parser = argparse.ArgumentParser(description="Distributed DPO Training")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--world_size", type=int, default=-1, help="Number of GPUs")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic training")
    args = parser.parse_args()
    
    # Print arguments for debugging
    if args.local_rank <= 0:  # Only print from rank 0 or if rank not set
        print(f"Arguments: {args}")
    
    # Get world size and rank
    if args.local_rank != -1:
        # When launched with torch.distributed.launch
        rank = args.local_rank
        world_size = torch.cuda.device_count() if args.world_size == -1 else args.world_size
    else:
        # Manual setup or single GPU
        rank = 0
        world_size = 1
    
    # Call main function
    main(rank, world_size, args)
