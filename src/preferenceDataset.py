from torch.utils.data import Dataset
from src.utils import format_input


# Prepare a custom PyTorch dataset
class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            prompt = format_input(entry)

            question_text = entry["question"]
            chosen_text = entry["chosen"]
            rejected_text = entry["rejected"]

            # Tokenize full texts
            chosen_full_text = f"{prompt}\n{chosen_text}{self.tokenizer.eos_token}"
            rejected_full_text = f"{prompt}\n{rejected_text}{self.tokenizer.eos_token}"

            # tokenize the full texts
            prompt_ids = self.tokenizer.encode(prompt)
            chosen_ids = self.tokenizer.encode(chosen_full_text)
            rejected_ids = self.tokenizer.encode(rejected_full_text)

            self.encoded_texts.append({
                "prompt": prompt_ids,
                "chosen": chosen_ids,
                "rejected": rejected_ids,
                "question_text": question_text,
                "chosen_text": chosen_text,
                "rejected_text": rejected_text
            })

        # print("\n=== data pre-processing validation ===")
        # sample_entry = self.encoded_texts[0]
        # print(f"Prompt:\n{self.tokenizer.decode(sample_entry['prompt'])}\n")
        # print(f"Chosen:\n{self.tokenizer.decode(sample_entry['chosen'])}\n")
        # print(f"Rejected:\n{self.tokenizer.decode(sample_entry['rejected'])}\n\n")
        # print(f"Question text:\n{sample_entry['question_text']}\n")
        # print(f"Chosen text:\n{sample_entry['chosen_text']}\n")
        # print(f"Rejected text:\n{sample_entry['rejected_text']}\n")
        # print("====================================\n")

    def __getitem__(self, index):
        item = self.encoded_texts[index]
        # print(f"Index: {index}, Prompt size: {len(item['prompt'])}, Chosen size: {len(item['chosen'])}, Rejected size: {len(item['rejected'])}")
        return item

    def __len__(self):
        return len(self.data)
