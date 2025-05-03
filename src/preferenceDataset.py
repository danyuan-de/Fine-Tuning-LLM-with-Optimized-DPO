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

            # Add EOS token to responses
            chosen_response = entry["chosen"] + self.tokenizer.eos_token
            rejected_response = entry["rejected"] + self.tokenizer.eos_token

            # Tokenize full texts
            chosen_full_text = f"{prompt}\n{chosen_response}"
            rejected_full_text = f"{prompt}\n{rejected_response}"

            # tokenize the full texts
            chosen_full_tokens = self.tokenizer.encode(chosen_full_text)
            rejected_full_tokens = self.tokenizer.encode(rejected_full_text)

            self.encoded_texts.append({
                "prompt": self.tokenizer.encode(prompt),
                "chosen": chosen_full_tokens,
                "rejected": rejected_full_tokens,
            })
        print("\n=== data pre-processing validation ===")
        sample_entry = self.encoded_texts[0]
        print("Prompt:", tokenizer.decode(sample_entry["prompt"]))
        print("Chosen:", tokenizer.decode(sample_entry["chosen"]))
        print("Rejected:", tokenizer.decode(sample_entry["rejected"]))

    def __getitem__(self, index):
        item = self.encoded_texts[index]
        # print(f"Index: {index}, Prompt size: {len(item['prompt'])}, Chosen size: {len(item['chosen'])}, Rejected size: {len(item['rejected'])}")
        return item

    def __len__(self):
        return len(self.data)
