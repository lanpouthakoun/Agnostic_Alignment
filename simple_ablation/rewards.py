import re
from vllm import SamplingParams
from utils.prompts import get_judge_prompt

def harmfulness_reward_func_vllm(prompts, completions, judge_llm=None, **kwargs):
    """
    Reward function using vLLM for fast batched inference.
    
    Args:
        prompts: List of input prompts
        completions: List of model completions/responses
        judge_llm: vLLM LLM instance for the judge model
        **kwargs: Additional arguments passed by GRPOTrainer
    
    Returns:
        List of reward scores (higher = less harmful, lower = more harmful)
    """
    if judge_llm is None:
        raise ValueError("judge_llm (vLLM instance) must be provided")
    
    # Prepare all prompts for batched inference
    formatted_prompts = [
        get_judge_prompt(prompt, completion)
        for prompt, completion in zip(prompts, completions)
    ]
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0,  # Greedy decoding (equivalent to do_sample=False)
        stop=["\n"],  # Optional: stop at newline for cleaner output
    )
    
    # Batch inference with vLLM - this is the key speedup
    outputs = judge_llm.generate(formatted_prompts, sampling_params)
    
    # Process outputs
    rewards = []
    for output, completion in zip(outputs, completions):
        generated_text = output.outputs[0].text.strip()
        
        # Extract score from response
        score_match = re.search(r'\b([1-5])\b', generated_text)
        if score_match:
            harmfulness_score = int(score_match.group(1))
        else:
            harmfulness_score = 3  # Default to neutral
        
        
        base_reward = (6 - harmfulness_score) / 5.0  # Range: 0.2 to 1.0
        
        
        
        rewards.append(base_reward)
        
    return rewards