import transformers
import torch
from torch import nn
from datasets import load_dataset

from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from rewards import harmfulness_reward_func
judge_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-2.5B")
judge_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-2.5B")
judge_model = judge_model.to('cuda')

dataset = load_dataset("parquet", data_files="simple_train.parquet")
test_dataset = load_dataset("parquet", data_files="simple_test.parquet")
dataset["train"] = dataset["train"].select(range(150))
test_dataset["train"] = test_dataset["train"].select(range(150))
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
    return harmfulness_reward_func(
        prompts, 
        completions, 
        judge_model_name="Qwen/Qwen3-2.5B",
        judge_model=judge_model,
        judge_tokenizer=judge_tokenizer,
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