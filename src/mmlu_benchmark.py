from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import src.config as config

# MMLU selected subjects
selected_subjects = [
    "high_school_physics",
    "college_physics",
    "econometrics",
    "global_facts",
    "formal_logic",
    "business_ethics"
]

ori_model_path = config.model_name
ori_tokenizer = AutoTokenizer.from_pretrained(ori_model_path)
ori_model = AutoModelForCausalLM.from_pretrained(ori_model_path)
ori_model.eval()

ft_model_path = config.fine_tuned_model_path
ft_tokenizer = AutoTokenizer.from_pretrained(ft_model_path)
ft_model = AutoModelForCausalLM.from_pretrained(ft_model_path)
ft_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ori_model.to(device)
ft_model.to(device)


def preprocess(example):
    question = example["question"]
    choices = example["choices"]
    inputs = [question + " " + choice for choice in choices]
    ori_encoding = ori_tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    ft_encoding = ft_tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    return ori_encoding, ft_encoding


def evaluate_subject(subject):
    print(f"Evaluating subject: {subject}")
    dataset = load_dataset(config.benchmark_dataset, subject, split="test")

    correct = 0
    total = 0

    for example in dataset:
        ori_inputs, ft_inputs = preprocess(example)
        ori_input_ids = ori_inputs["input_ids"].view(1, -1, ori_inputs["input_ids"].size(-1)).to(device)
        ft_input_ids = ft_inputs["input_ids"].view(1, -1, ft_inputs["input_ids"].size(-1)).to(device)
        ori_attention_mask = ori_inputs["attention_mask"].view(1, -1, ori_inputs["attention_mask"].size(-1)).to(device)
        attention_mask = ft_inputs["attention_mask"].view(1, -1, ft_inputs["attention_mask"].size(-1)).to(device)

        with torch.no_grad():
            ori_outputs = ori_model(input_ids=ori_input_ids, attention_mask=ori_attention_mask)
            ori_logits = ori_outputs.logits
            ft_outputs = ft_model(input_ids=ft_input_ids, attention_mask=attention_mask)
            ft_logits = ft_outputs.logits

        ori_prediction = torch.softmax(ori_logits, dim=-1)
        ft_prediction = torch.softmax(ft_logits, dim=-1)
        if ori_prediction == example["answer"]:
            ori_correct += 1
        if ft_prediction == example["answer"]:
            ft_correct += 1
        total += 1

    ori_accuracy = ori_correct / total
    ft_accuracy = ft_correct / total
    print(f"{subject}: Original Model Accuracy: {ori_accuracy:.2%}")
    print(f"{subject}: Fine-tuned Model Accuracy: {ft_accuracy:.2%}")
    return subject, ori_accuracy, ft_accuracy


# Run evaluation for each subject
results = {}
for subject in selected_subjects:
    subject, ori_acc, ft_acc = evaluate_subject(subject)
    results[subject]["original"] = ori_acc
    results[subject]["fine_tuned"] = ft_acc

# print the summary of results
# store the results in a json file
print("\n=== Summary ===")
for k, v in results.items():
    print(f"{k}: {v:.2%}")
