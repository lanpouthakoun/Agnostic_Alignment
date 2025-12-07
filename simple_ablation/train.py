import transformers
import torch
from torch import nn
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from rewards import harmfulness_reward_func_vllm

# Initialize vLLM for the judge model (much faster inference)
judge_llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=1,  # Increase if you have multiple GPUs
    gpu_memory_utilization=0.05,  # Adjust based on your GPU memory
    max_model_len=2048,
)

dataset = load_dataset("parquet", data_files="data/simple_train.parquet")
test_dataset = load_dataset("parquet", data_files="data/simple_test.parquet")
dataset["train"] = dataset["train"].select(range(400))
test_dataset["train"] = test_dataset["train"].select(range(400))

print(dataset["train"][0])  # Inspect a few examples


config = GRPOConfig(
    log_completions=True,
    num_completions_to_print=5,
    logging_steps=10,
    output_dir="./grpo_output",
    num_train_epochs=1,
    save_strategy="steps",
    save_total_limit=10,
    save_on_each_node=True,
    per_device_train_batch_size=8,
    use_vllm = False
)

def refusal_reward_with_vllm(prompts, completions, **kwargs):
    return harmfulness_reward_func_vllm(
        prompts,
        completions,
        judge_llm=judge_llm,
        **kwargs
    )

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-7B",
    reward_funcs=[refusal_reward_with_vllm],
    args=config,
    train_dataset=dataset["train"],
    eval_dataset=test_dataset["train"],
)

trainer.train()