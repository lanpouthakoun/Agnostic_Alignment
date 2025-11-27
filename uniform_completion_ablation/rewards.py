import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.prompts import get_refusal_prompt
import re


def refusal_reward(prompts, completions, refusal_classifier=None, refusal_tokenizer=None, refusal_model_name="Qwen/Qwen3-0.6B", device=None, **kwargs):
    """
    Reward function for GRPOTrainer that evaluates whether a model immediately refused.
    
    Args:
        prompts: List of input prompts
        completions: List of model completions/responses
        refusal_classifier: Pre-loaded refusal classifier model (optional)
        refusal_tokenizer: Pre-loaded refusal tokenizer (optional)
        refusal_model_name: Name of the refusal model to load (optional, can use judge_model_name instead)
        judge_model_name: Name of the judge model to load (passed by GRPOTrainer, used as alias for refusal_model_name)
        device: Device to run inference on (defaults to model's device)
        **kwargs: Additional arguments passed by GRPOTrainer
    
    Returns:
        List of reward scores (higher = less harmful, lower = more harmful)
    """
    # Accept judge_model_name from GRPOTrainer as an alias for refusal_model_name
    if refusal_model_name is None:
        raise ValueError("Either provide refusal classifier or specify refusal_model_name (or judge_model_name)")
    
    if refusal_classifier is None:
        if refusal_model_name is None:
            raise ValueError("If refusal classifier is provided without refusal_model_name, specify refusal_model_name")
        
        if refusal_tokenizer is None:
            refusal_tokenizer = AutoTokenizer.from_pretrained(refusal_model_name)
        refusal_classifier = AutoModelForCausalLM.from_pretrained(refusal_model_name)
    elif refusal_tokenizer is None:
        if refusal_model_name is None:
            raise ValueError("If refusal model is provided without refusal_tokenizer, specify refusal_model_name (or judge_model_name)")
        refusal_tokenizer = AutoTokenizer.from_pretrained(refusal_model_name)
    
    if device is None:
        device = next(refusal_classifier.parameters()).device
    refusal_classifier = refusal_classifier.to(device)
    refusal_classifier.eval()
    
    rewards = []
    
    with torch.no_grad():
        for prompt, completion in zip(prompts, completions):
            formatted_prompt = get_refusal_prompt(completion)
            
            inputs = refusal_tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(device)
            
            
            pad_token_id = refusal_tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = refusal_tokenizer.eos_token_id
            
            outputs = refusal_classifier.generate(
                inputs.input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=refusal_tokenizer.eos_token_id,
            )
            
            # Decode the response
            generated_text = refusal_tokenizer.decode(
                outputs[0][inputs.input_ids.shape[-1]:],
                skip_special_tokens=True
            ).strip()
            
            score_match = re.search(r'\b([1-5])\b', generated_text)
            if score_match:
                reward = int(score_match.group(1))
            else:
                reward = 0
            
            rewards.append(reward)
    
    return rewards
