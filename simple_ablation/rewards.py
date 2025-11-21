import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.prompts import get_judge_prompt
import re


def harmfulness_reward_func(prompts, completions, judge_model=None, judge_tokenizer=None, judge_model_name=None, device=None, **kwargs):
    """
    Reward function for GRPOTrainer that evaluates harmfulness of completions.
    
    Args:
        prompts: List of input prompts
        completions: List of model completions/responses
        judge_model: Pre-loaded judge model (optional, will load if not provided)
        judge_tokenizer: Pre-loaded judge tokenizer (optional, will load if not provided)
        judge_model_name: Name of the judge model to load (required if model/tokenizer not provided)
        device: Device to run inference on (defaults to model's device or 'cuda' if available)
        **kwargs: Additional arguments passed by GRPOTrainer
    
    Returns:
        List of reward scores (higher = less harmful, lower = more harmful)
    """
    if judge_model is None:
        if judge_model_name is None:
            raise ValueError("Either provide judge_model or specify judge_model_name")
        
        if judge_tokenizer is None:
            judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
        judge_model = AutoModelForCausalLM.from_pretrained(judge_model_name)
    elif judge_tokenizer is None:
        if judge_model_name is None:
            raise ValueError("If judge_model is provided without judge_tokenizer, specify judge_model_name")
        judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
    
    if device is None:
        device = next(judge_model.parameters()).device
    judge_model = judge_model.to(device)
    judge_model.eval()
    
    rewards = []
    
    with torch.no_grad():
        for prompt, completion in zip(prompts, completions):
            formatted_prompt = get_judge_prompt(prompt, completion)
            
            inputs = judge_tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(device)
            
            
            pad_token_id = judge_tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = judge_tokenizer.eos_token_id
            
            outputs = judge_model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=judge_tokenizer.eos_token_id,
            )
            
            # Decode the response
            generated_text = judge_tokenizer.decode(
                outputs[0][inputs.input_ids.shape[-1]:],
                skip_special_tokens=True
            ).strip()
            
            score_match = re.search(r'\b([1-5])\b', generated_text)
            if score_match:
                harmfulness_score = int(score_match.group(1))
            else:
                harmfulness_score = 3
            
            reward = (6 - harmfulness_score) / 5.0
            rewards.append(reward)
    
    return rewards
