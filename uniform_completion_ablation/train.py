import torch
from torch import nn
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from rewards import refusal_reward
from transformers import AutoTokenizer, AutoModelForCausalLM
dataset = load_dataset("parquet", data_files="hh_rlhf_single_turn_train.parquet")
test_dataset = load_dataset("parquet", data_files="hh_rlhf_single_turn_test.parquet")
dataset["train"] = dataset["train"].select(range(150))
test_dataset["train"] = test_dataset["train"].select(range(150))
print(test_dataset.keys())
refusal_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
refusal_classifier = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
refusal_classifier = refusal_classifier.to('cuda')

config = GRPOConfig(
    log_completions=True,
    num_completions_to_print=5,
    logging_steps=10,
    output_dir="./grpo_output",
    num_train_epochs=1,
    save_strategy="steps",
    save_total_limit=10,  # Keep last 10 checkpoints
    save_on_each_node=True,
    per_device_train_batch_size = 8
)

def refusal_reward_with_model(prompts, completions, **kwargs):
    return refusal_reward(
        prompts, 
        completions, 
        refusal_model_name="Qwen/Qwen3-0.6B",
        refusal_tokenizer=refusal_tokenizer,
        refusal_classifier=refusal_classifier,
        device='cuda',
        **kwargs
    )

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-3B-Instruct",
    reward_funcs=[refusal_reward_with_model],
    args=config,
    train_dataset=dataset["train"],
    eval_dataset=test_dataset["train"],
)

trainer.train()
# Explicitly save the final model
trainer.save_model()