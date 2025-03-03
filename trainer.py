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

training_args = DPOConfig(output_dir=f"{model_name}_DPO", logging_steps=10)
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()

trainer.save_model()
trainer.save_discriminator()

# Evaluate the model
val_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="validation")
trainer.evaluate(val_dataset)
# print the first 3 examples in validation dataset
for example in val_dataset["text"][:3]:
    print(f"Example: {example}")
    print(f"Generated text: {trainer.generate_text(example)}")
