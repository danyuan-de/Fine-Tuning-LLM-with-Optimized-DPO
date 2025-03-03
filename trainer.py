# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import config

model_name = config.model_name
cache_dir = config.cache_dir

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

training_args = DPOConfig(output_dir=f"{model_name}_DPO", logging_steps=5, per_device_train_batch_size=config.batch_size, num_train_epochs=config.num_epochs, learning_rate=config.learning_rate, beta=config.beta, temperature=config.temperature, top_p=config.top_p, max_new_tokens=config.max_new_tokens, allowed_max_length=config.allowed_max_length)
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()


trainer.save_model()


# Evaluate the model
val_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="validation")
trainer.evaluate(val_dataset)
# print the first 3 examples in validation dataset
for example in val_dataset["text"][:3]:
    print(f"Example: {example}")
    print(f"Generated text: {trainer.generate_text(example)}")

    # print expected chosen and rejected text
    print(f"Expected chosen text: {example['chosen_text']}")
    print(f"Expected rejected text: {example['rejected_text']}")

# load fine-tuned model after training
ft_model = AutoModelForCausalLM.from_pretrained(f"{model_name}_DPO")
ft_tokenizer = AutoTokenizer.from_pretrained(f"{model_name}_DPO")

# generate text using fine-tuned model from the first 3 examples in test dataset
test_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="test")

for example in test_dataset["text"][:3]:
    print(f"Example: {example}")
    print(f"Generated text: {trainer.generate_text(example)}")

    # print expected chosen and rejected text
    print(f"Expected chosen text: {example['chosen_text']}")
    print(f"Expected rejected text: {example['rejected_text']}")