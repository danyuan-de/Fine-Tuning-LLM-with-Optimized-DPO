# src/test_evaluator.py
import json
from tqdm import tqdm
from datetime import timedelta
import torch
import time
from src.utils import format_input, text_to_token_ids, postprocess_response, calculate_perplexity, generate
import src.config as config


def save_test_results(test_results, output_json_path):
    """
    Save the test results to a JSON file.
    """
    with open(output_json_path, "w") as f:
        json.dump(test_results, f, indent=4)
    print("Test results saved to:", output_json_path)


def test_and_evaluate_one(
    output_json_path: str,
    test_data,
    test_loader,
    dpo_loss_fn,
    ref_model,
    ref_tokenizer,
    fine_tuned_model,
    fine_tuned_tokenizer,
    device,
    eval_temperature,
    eval_top_p,
    max_new_tokens,
    stride_length,
    eos_token_id
):
    # Check first entry to determine data type
    input_key = "question" if "question" in test_data[0] else "instruction"

    print("Starting test evaluation...")
    # Evaluate the model on the test set
    test_res = dpo_loss_fn.evaluate_dpo_loss_loader(
        policy_model=fine_tuned_model,
        reference_model=ref_model,
        train_loader=None,
        val_loader=test_loader,
        eval_iter=5
    )
    print("Test loss:", test_res["val_loss"])
    print("Test reward margin:", test_res["val_chosen_reward"] - test_res["val_rejected_reward"])

    test_results = []
    test_start_time = time.time()
    try:
        for i, entry in enumerate(test_data):

            input_text = format_input(entry)

            # Reference Model Generation
            ref_input_ids = text_to_token_ids(input_text, ref_tokenizer).to(device)
            ref_generated = generate(
                model=ref_model,
                idx=ref_input_ids.to(device),
                max_new_tokens=max_new_tokens,
                temperature=eval_temperature,
                top_p=eval_top_p,
                eos_token_id=eos_token_id
            )
            ref_full_text = ref_tokenizer.decode(ref_generated[0], skip_special_tokens=False)
            ref_response = postprocess_response(ref_full_text)

            # Fine-Tuned Model Generation
            fine_tuned_model_input_ids = text_to_token_ids(input_text, fine_tuned_tokenizer).to(device)
            fine_tuned_model_generated = generate(
                model=fine_tuned_model,
                idx=fine_tuned_model_input_ids.to(device),
                max_new_tokens=max_new_tokens,
                temperature=eval_temperature,
                top_p=eval_top_p,
                eos_token_id=eos_token_id
            )
            fine_tuned_model_full_text = fine_tuned_tokenizer.decode(fine_tuned_model_generated[0], skip_special_tokens=False)
            fine_tuned_model_response = postprocess_response(fine_tuned_model_full_text)

            # Calculate perplexity
            ref_perplexity = calculate_perplexity(
                model=ref_model,
                tokenizer=ref_tokenizer,
                texts=input_text,
                max_length=config.allowed_max_length,
                stride=stride_length,
                device=device
            )

            ft_perplexity = calculate_perplexity(
                model=fine_tuned_model,
                tokenizer=fine_tuned_tokenizer,
                texts=input_text,
                max_length=config.allowed_max_length,
                stride=stride_length,
                device=device
            )

            # Use the previously determined input key
            print(f"\nInput {i}:\n {entry[input_key]}")

            print("\n ----- Reference Model ----- ")
            print(f"Reference Response: {ref_response}")
            print(f"Perplexity: {ref_perplexity:.2f}")

            print("\n ----- Policy Model ----- ")
            print(f"Policy Response: {fine_tuned_model_response}")
            print(f"Perplexity: {ft_perplexity:.2f}")

            print("\n ----- Expected Response ----- ")
            print(f"Expected Answer: {entry['chosen']}")
            print("=" * 80, "\n")

            # Create a single sample object and append to the results list
            sample = {
                input_key: entry[input_key],
                "ref_response": ref_response,
                "policy_response": fine_tuned_model_response,
                "expected_response": entry['chosen'],
                "ref_perplexity": ref_perplexity,
                "policy_perplexity": ft_perplexity
            }
            test_results.append(sample)

        test_end_time = time.time()
        test_execution_time = (test_end_time - test_start_time) / 60
        print(f"Test evaluation completed in {test_execution_time:.2f} minutes (in {str(timedelta(seconds=test_end_time - test_start_time))})")

    except KeyboardInterrupt:
        print("\nInterrupted! Saving partial results...")

    finally:
        save_test_results(test_results, output_json_path)
        print("Test results saved to:", output_json_path)


def test_and_evaluate_batch(
    output_json_path: str,
    test_data,
    test_loader,
    dpo_loss_fn,
    ref_model,
    ref_tokenizer,
    fine_tuned_model,
    fine_tuned_tokenizer,
    device,
    eval_temperature,
    eval_top_p,
    max_new_tokens,
    stride_length,
    eos_token_id
):
    """
    Evaluate the model on the test set and save the results to a JSON file.
    """

    # Check first entry to determine data type
    input_key = "question" if "question" in test_data[0] else "instruction"

    print("Starting test evaluation...")
    # Evaluate the model on the test set
    test_res = dpo_loss_fn.evaluate_dpo_loss_loader(
        policy_model=fine_tuned_model,
        reference_model=ref_model,
        train_loader=None,
        val_loader=test_loader,
        eval_iter=5
    )
    print("Test loss:", test_res["val_loss"])
    print("Test reward margin:", test_res["val_chosen_reward"] - test_res["val_rejected_reward"])

    try:
        test_results = []
        test_start_time = time.time()
        for count, batch in enumerate(tqdm(test_loader, desc="Test Eval")):
            questions = batch["question_texts"]   # list of str, length = batch_size
            expected = batch["chosen_texts"]
            denied = batch["rejected_texts"]

            # batch["prompt"] is a tensor of token ids (bsz, seq_len)
            # 1) decoding prompt
            start = count * config.test_batch_size
            end = start + len(batch["question_texts"])
            original_test_data_slice = test_data[start:end]
            full_prompts = [format_input(entry) for entry in original_test_data_slice]

            # 2) Batch generation of responses using the reference and fine-tuned models
            ref_input_ids = text_to_token_ids(full_prompts, ref_tokenizer).to(device)
            pol_input_ids = text_to_token_ids(full_prompts, fine_tuned_tokenizer).to(device)
            with torch.no_grad():
                ref_out = generate(
                    model=ref_model,
                    idx=ref_input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=eval_temperature,
                    top_p=eval_top_p,
                    eos_token_id=eos_token_id
                )
                pol_out = generate(
                    model=fine_tuned_model,
                    idx=pol_input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=eval_temperature,
                    top_p=eval_top_p,
                    eos_token_id=eos_token_id
                )

            # 3) Batch decoding the generated responses
            ref_resps = [postprocess_response(ref_tokenizer.decode(ids, skip_special_tokens=False)) for ids in ref_out]
            pol_resps = [postprocess_response(fine_tuned_tokenizer.decode(ids, skip_special_tokens=False)) for ids in pol_out]

            ref_texts = [f"{p}{r}{ref_tokenizer.eos_token}" for p, r in zip(full_prompts, ref_resps)]
            pol_texts = [f"{p}{r}{fine_tuned_tokenizer.eos_token}" for p, r in zip(full_prompts, pol_resps)]

            ref_exp_texts = [f"{p}{r}{ref_tokenizer.eos_token}" for p, r in zip(full_prompts, expected)]
            ref_denied_texts = [f"{p}{r}{ref_tokenizer.eos_token}" for p, r in zip(full_prompts, denied)]
            pol_exp_texts = [f"{p}{r}{fine_tuned_tokenizer.eos_token}" for p, r in zip(full_prompts, expected)]
            pol_denied_texts = [f"{p}{r}{fine_tuned_tokenizer.eos_token}" for p, r in zip(full_prompts, denied)]

            # 4) Calculate perplexity for the reference and fine-tuned models and store them
            ref_ppls = calculate_perplexity(ref_model, ref_tokenizer, ref_texts, max_length=config.allowed_max_length, stride=stride_length, batch_size=config.test_batch_size, device=device)
            pol_ppls = calculate_perplexity(fine_tuned_model, fine_tuned_tokenizer, pol_texts, max_length=config.allowed_max_length, stride=stride_length, batch_size=config.test_batch_size, device=device)
            ref_exp_ppls = calculate_perplexity(ref_model, ref_tokenizer, ref_exp_texts, max_length=config.allowed_max_length, stride=stride_length, batch_size=config.test_batch_size, device=device)
            pol_exp_ppls = calculate_perplexity(fine_tuned_model, fine_tuned_tokenizer, pol_exp_texts, max_length=config.allowed_max_length, stride=stride_length, batch_size=config.test_batch_size, device=device)
            ref_denied_ppls = calculate_perplexity(ref_model, ref_tokenizer, ref_denied_texts, max_length=config.allowed_max_length, stride=stride_length, batch_size=config.test_batch_size, device=device)
            pol_denied_ppls = calculate_perplexity(fine_tuned_model, fine_tuned_tokenizer, pol_denied_texts, max_length=config.allowed_max_length, stride=stride_length, batch_size=config.test_batch_size, device=device)

            if not isinstance(ref_ppls, list):
                ref_ppls = [ref_ppls]
            if not isinstance(pol_ppls, list):
                pol_ppls = [pol_ppls]
            if not isinstance(ref_exp_ppls, list):
                ref_exp_ppls = [ref_exp_ppls]
            if not isinstance(pol_exp_ppls, list):
                pol_exp_ppls = [pol_exp_ppls]
            if not isinstance(ref_denied_ppls, list):
                ref_denied_ppls = [ref_denied_ppls]
            if not isinstance(pol_denied_ppls, list):
                pol_denied_ppls = [pol_denied_ppls]

            # 5) Print the results and store them in the dictionary
            for i, question in enumerate(questions):
                # Use the previously determined input key
                print(f"\nInput {i + (count * 5) + 1}:\n {question}")

                print("\n ----- Reference Model ----- ")
                print(f"Reference Response:\n {ref_resps[i]}")
                print(f"Perplexity: {ref_ppls[i]:.2f}")

                print("\n ----- Policy Model ----- ")
                print(f"Policy Response:\n {pol_resps[i]}")
                print(f"Perplexity: {pol_ppls[i]:.2f}")

                print("\n ----- Expected Response ----- ")
                print(f"Expected Answer:\n {expected[i]}")
                print(f"Gold Answer PPL (ref):    {ref_exp_ppls[i]:.2f}")
                print(f"Disliked Answer PPL (ref): {ref_denied_ppls[i]:.2f}")
                print(f"Gold Answer PPL (policy): {pol_exp_ppls[i]:.2f}")
                print(f"Disliked Answer PPL (policy): {pol_denied_ppls[i]:.2f}")
                print("=" * 80, "\n")

                test_results.append({
                    input_key: question,
                    "ref_response": ref_resps[i],
                    "policy_response": pol_resps[i],
                    "expected": expected[i],
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
        save_test_results(test_results, output_json_path)
        print("Test results saved to:", output_json_path)
