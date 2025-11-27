import torch
from torch import nn
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from rewards import refusal_reward
from transformers import AutoTokenizer, AutoModelForCausalLM
dataset = load_dataset("parquet", data_files="hh_rlhf_single_turn_train.parquet")
test_dataset = load_dataset("parquet", data_files="hh_rlhf_single_turn_test.parquet")
print(test_dataset.keys())
refusal_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
refusal_classifier = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
refusal_classifier = refusal_classifier.to('cuda')
training_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
training_model = training_model.to('cuda')
training_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
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
    model=training_model,
    tokenizer=training_tokenizer,
    reward_funcs=[refusal_reward_with_model],
    args=GRPOConfig(
        output_dir="./grpo_output",
        num_train_epochs=1,
        save_strategy="steps",
        save_steps=1,  # Save after every step
        save_total_limit=10,  # Keep last 10 checkpoints
        save_on_each_node=True,
    ),
    train_dataset=dataset["train"],
    eval_dataset=test_dataset["train"],
)

trainer.train()
# Explicitly save the final model
trainer.save_model()