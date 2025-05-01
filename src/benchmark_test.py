import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import time
from datetime import timedelta
import random

import src.config as config
from src.utils import postprocess_response, format_input


def run_benchmark():
    # Load the dataset based on config
    if config.benchmark_dataset == config.benchmark_datasets[1]:
        print("Running benchmark on: PHYBench")
        ds = load_dataset("Eureka-Lab/PHYBench", split="train")  # use 'train' split
        examples = ds
    elif config.benchmark_dataset == config.benchmark_datasets[2]:
        print("Running benchmark on: MMLU-Pro")
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")  # use 'test' split
        if config.MMLU_PRO_category_isPhysics:
            # Filter to physics category only
            examples = ds.filter(lambda x: x['category'] == 'physics')
        else:
            examples = ds
    else:
        raise ValueError(f"Unsupported benchmark dataset id: {config.benchmark_dataset}")

    print(f"Loaded {len(examples)} examples from the dataset.")

    random.seed(42)  # Set seed for reproducibility
    random.shuffle(examples)  # Shuffle the dataset
    if config.num_benchmark_samples > 0:
        examples = examples[:config.num_benchmark_samples]
        print(f"Using {len(examples)} samples for benchmarking.")
    else:
        print("Using all samples for benchmarking.")

    try:
        # Load the fine-tuned model
        model_path = config.fine_tuned_model_path
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device).eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # Use sampling or greedy decoding for evaluation
    if config.EVAL_USE_SAMPLING:
        print("Using sampling for evaluation")
        eval_temperature = config.temperature
        eval_top_p = config.top_p
    else:
        print("Using greedy decoding for evaluation")
        eval_temperature = 0.0
        eval_top_p = None

    records = []
    start_time = time.time()

    for i, entry in enumerate(examples):

        prompt = format_input(entry)
        if prompt is None:
            print(f"Skipping entry with id {entry.get('id', entry.get('question_id', 'unknown'))}: format_input returned None.")
            continue

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=1024,
                temperature=eval_temperature,
                top_p=eval_top_p,
                eos_token_id=tokenizer.eos_token_id
            )
        raw = tokenizer.decode(out_ids[0], skip_special_tokens=False)
        raw_resp = postprocess_response(raw)

        record = {
            'id': entry.get('id') or entry.get('question_id'),
            'tag': entry.get('tag') or entry.get('category'),
            'prompt': entry.get('content') or entry.get('question'),
            'options': entry.get('options', []),
            'raw_output': raw_resp,
            'expected_answer': entry.get('answer', ''),
        }

        # for PHYBench, add expected_solution
        if config.benchmark_dataset == config.benchmark_datasets[1]:
            record['expected_solution'] = entry.get('solution', '')

        # for MMLU-Pro, add options, answer_index, and chain-of-thought
        if config.benchmark_dataset == config.benchmark_datasets[2]:
            record['answer_index'] = entry.get('answer_index', -1)
            record['cot'] = entry.get('cot', '')  # chain-of-thought
            record['src'] = entry.get('src', '')
        records.append(record)

        # Print the record for debugging
        print(f"\nRecord {i + 1}, in percentage: {(i + 1) / len(examples) * 100:.2f}% :")
        print(f"\nID: {record['id']}")
        print(f"Tag: {record['tag']}")
        print(f"Prompt: {record['prompt']}")
        print(f"Options: {record['options']}")
        print(f"Raw Output: {record['raw_output']}")
        print(f"Expected Answer: {record['expected_answer']}")
        if config.benchmark_dataset == config.benchmark_datasets[2]:
            print(f"answer_index: {record['answer_index']}")

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Benchmark completed in {execution_time_minutes:.2f} minutes (in {str(timedelta(seconds=end_time - start_time))})")

    try:
        # Save the records to a JSON file
        os.makedirs(config.result_dir, exist_ok=True)

        training_dtype = next(
            (dtype for dtype in ["content", "mixed", "html", "structure", "preference"]
             if dtype in config.training_data_filename),
            "unknown"
        )
        benchmark_short = config.benchmark_dataset.split('/')[-1]
        if config.benchmark_dataset == config.benchmark_datasets[2] and config.MMLU_PRO_category_isPhysics:
            benchmark_short += "_Physics"

        benchmark_name = f'benchmark_{benchmark_short}_{training_dtype}_{config.method_name}.json'

        detail_path = os.path.join(config.result_dir, benchmark_name)
        with open(detail_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"Saved benchmark records to {benchmark_name}")

    except Exception as e:
        print(f"Error saving benchmark records: {e}")
        raise
