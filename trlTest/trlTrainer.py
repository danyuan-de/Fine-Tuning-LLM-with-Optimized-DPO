# train_dpo.py
from datasets import load_dataset, Dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import src.config as config
import json
from utility import format_input

model_name = config.model_name
cache_dir = config.cache_dir
save_path = config.fine_tuned_model_path
file_path = f"../{config.file_content}"

# Load your custom JSON data
def load_custom_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Convert to a format compatible with DPO: each entry needs 'prompt', 'chosen', and 'rejected'
    formatted_data = {
        "prompt": [entry["question"] for entry in data],
        "chosen": [entry["chosen"] for entry in data],
        "rejected": [entry["rejected"] for entry in data]
    }
    # Create a Dataset object
    return Dataset.from_dict(formatted_data)

dataset = load_custom_dataset(file_path)

# Split the dataset into train and test (e.g., 80% train, 20% test)
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
# train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# # Use 1/20 of the training dataset
# train_dataset = train_dataset.shard(num_shards=20, index=0)

eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
print("This is the chat template")
print(tokenizer.chat_template)
if tokenizer.chat_template is None:
    tokenizer.chat_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id>\n{input}<|eot_id|>"

print("\n -------------- \n")
print("Start training")
training_args = DPOConfig(
    output_dir=f"{model_name}_DPO",
    logging_steps=5,
    per_device_train_batch_size=config.batch_size,
    num_train_epochs=config.num_epochs,
    beta=config.beta,
    padding_value=eot_token_id
)
trainer = DPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset
)
trainer.train()
print("Training finished")

print("Saving Model")
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print("\n -------------- \n")

print("Start evaluation")

ft_model = AutoModelForCausalLM.from_pretrained(save_path)
ft_tokenizer = AutoTokenizer.from_pretrained(save_path)

# Load test dataset
# test_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="test")

# Generate text for the first 3 examples
for example in test_dataset[:3]:
    input_text = format_input(example)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

    # Generate text with original model
    original_output = model.generate(
        **inputs,
        max_new_tokens=50,  # Adjust as needed
        do_sample=True,     # Optional: for varied outputs
        pad_token_id=eot_token_id
    )
    original_text = tokenizer.decode(original_output[0], skip_special_tokens=True)

    # Generate text with fine-tuned model
    ft_output = ft_model.generate(
        **inputs,
        max_new_tokens=50,  # Adjust as needed
        do_sample=True,     # Optional: for varied outputs
        pad_token_id=eot_token_id
    )
    ft_text = ft_tokenizer.decode(ft_output[0], skip_special_tokens=True)

    # Print results
    print(f"Example prompt: {input_text}")
    print(f"Generated text by original model: {original_text}")
    print(f"Generated text by fine-tuned model: {ft_text}")
    print(f"Expected chosen text: {example['chosen']}")
    print(f"Expected rejected text: {example['rejected']}")
    print("\n")
