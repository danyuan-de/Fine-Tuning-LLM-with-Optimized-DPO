import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import numpy as np
import src.config as config
import json


def load_model(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.eval().to(device)

    model.config.do_sample = False  # greedy
    model.config.temperature = 1.0
    model.config.top_p = 1.0
    return tokenizer, model


# -------- PROMPT & SCORING UTILS --------
def format_mmlu_prompt(subject: str, question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{chr(65+i)}) {c.strip()}" for i, c in enumerate(choices))
    return (
        f"You are an expert in {subject}.\n\n"
        f"Multiple‐choice question:\n{question.strip()}\n\n"
        f"{opts}\n\n"
        "Please choose exactly one option and output ONLY the letter (A, B, C, …) of your answer."
    )


def score_choices(model, tokenizer, question: str, choices: list[str], device) -> list[float]:
    """
    Score the choices using the model.
    Args:
        model: The model to use for scoring.
        tokenizer: The tokenizer to use for encoding the input.
        question: The question to ask.
        choices: The list of choices to score.
    Returns:
        A list of scores for each choice.
    """
    scores = []
    for choice in choices:
        text = question + " " + choice
        encoded = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoded, labels=encoded["input_ids"])
            # outputs.loss is the average negative log likelihood, equivalent to -mean(log p(x_t | x_<t))
            # Multiply it by the sequence length to get the total negative log likelihood:
            neg_log_likelihood = outputs.loss * encoded["input_ids"].size(1)
        scores.append(-neg_log_likelihood.item())  # higher is better
    return scores


def evaluate_subject(subject, ori_tokenizer, ori_model, ft_tokenizer, ft_model, device):
    print(f"Evaluating subject: {subject}")
    ds = load_dataset(BENCHMARK, subject, split="test")
    rows = []

    ori_correct = 0
    ft_correct = 0
    total = 0

    for ex in ds:
        question = ex["question"]
        choices = ex["choices"]
        answer = ex["answer"]  # Assuming this is a 0-based index

        # Score the choices using both models
        ori_scores = score_choices(ori_model, ori_tokenizer, question, choices, device)
        ft_scores = score_choices(ft_model, ft_tokenizer, question, choices, device)

        # Take the higher score for each model
        ori_pred = int(np.argmax(ori_scores))
        ft_pred = int(np.argmax(ft_scores))

        # Single token generation
        prompt = format_mmlu_prompt(subject, question, choices)

        # original
        inp_ori = ori_tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ori = ori_model.generate(
                **inp_ori,
                max_new_tokens=1,
                do_sample=False
            )
        ori_letter = ori_tokenizer.decode(
            out_ori[0, inp_ori.input_ids.shape[-1]:]
        ).strip()

        # fine‐tuned
        inp_ft = ft_tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ft = ft_model.generate(
                **inp_ft,
                max_new_tokens=1,
                do_sample=False
            )
        ft_letter = ft_tokenizer.decode(
            out_ft[0, inp_ft.input_ids.shape[-1]:]
        ).strip()

        if ori_pred == answer:
            ori_correct += 1

        if ft_pred == answer:
            ft_correct += 1
        total += 1

        # build row dict
        rows.append({
            "question": question,
            "answer_idx": answer,
            "ori_pred_idx": ori_pred,
            "ft_pred_idx": ft_pred,
            "ori_pred_letter": ori_letter,
            "ft_pred_letter": ft_letter,
            "ori_scores": {chr(65 + i): s for i, s in enumerate(ori_scores)},
            "ft_scores": {chr(65 + i): s for i, s in enumerate(ft_scores)},
        })

    ori_acc = ori_correct / total
    ft_acc = ft_correct / total
    print(f"{subject}: Original Model Accuracy: {ori_acc:.2%}")
    print(f"{subject}: Fine-tuned Model Accuracy: {ft_acc:.2%}")
    return {
        "examples": rows,
        "ori_acc": ori_acc,
        "ft_acc": ft_acc
    }


if __name__ == "__main__":
    BENCHMARK = "cais/mmlu"
    SUBJECTS = [
        "high_school_physics",
        "college_physics",
        "econometrics",
        "global_facts",
        "formal_logic",
        "business_ethics",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading models...")
    print(f"The original model is: {config.model_name}")
    # Load the original and fine-tuned models
    ori_tokenizer, ori_model = load_model(config.model_name, device)
    # Load the fine-tuned model
    ft_tokenizer, ft_model = load_model(config.fine_tuned_model_path, device)

    # # Single‐token, greedy gen config
    # ori_gen_cfg = GenerationConfig(
    #     max_new_tokens=1,
    #     do_sample=False,
    #     temperature=1.0,
    #     top_p=1.0,
    #     top_k=1,
    #     bos_token_id=ori_tokenizer.bos_token_id,
    #     pad_token_id=ori_tokenizer.pad_token_id,
    #     eos_token_id=ori_tokenizer.eos_token_id
    # )
    # ft_gen_cfg = GenerationConfig(
    #     max_new_tokens=1,
    #     do_sample=False,
    #     temperature=1.0,
    #     top_p=1.0,
    #     top_k=1,
    #     bos_token_id=ft_tokenizer.bos_token_id,
    #     pad_token_id=ft_tokenizer.pad_token_id,
    #     eos_token_id=ft_tokenizer.eos_token_id
    # )

    all_results = {}
    for subj in SUBJECTS:
        print(f"Evaluating {subj} …")
        all_results[subj] = evaluate_subject(
            subj,
            ori_tokenizer, ori_model,
            ft_tokenizer, ft_model,
            device
        )

    out_path = os.path.join(config.result_dir, "mmlu_all_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to: {out_path}")
