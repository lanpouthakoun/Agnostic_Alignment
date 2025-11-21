import transformers
import torch
from torch import nn
from trl import GRPOTrainer
from transformers import load_dataset
from rewards import harmfulness_reward_func

dataset = load_dataset("json", data_files="datasets/harmfulness_dataset.json")

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-3B-Instruct",
    reward_funcs=[harmfulness_reward_func],
    judge_model_name="Qwen/Qwen2.5-3B-Instruct",
    train_dataset = dataset["train"],
    eval_dataset = dataset["eval"],
)

trainer.train()