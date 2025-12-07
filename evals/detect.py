"""
This file uses LLM as a Judge to Evaluate whether a refusal was made or not.

Looks just at the response from the model

"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
# from prompts.prompt import safety_prompt
from prompts.prompt import safety_prompt

# base_instruction = safety_prompt()
base_instruction = ""
print(f"{base_instruction } this is the prompt")
class JudgeVLLM:
    def __init__(
        self,
        judge_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.4,
        chat_template: str = None, 
        instruction: str = base_instruction
    ):
        # Remove this line:
        # self.model = AutoModelForCausalLM.from_pretrained(judge_model)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            judge_model,
            trust_remote_code=True,
        )
        self.instruction = instruction
        if chat_template:
            self.tokenizer.chat_template = chat_template
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm = LLM(
            model=judge_model,  # Pass the string path, not a model object
            tokenizer=judge_model,  # Same here - use the string path
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=4096,
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            stop_token_ids=[self.tokenizer.eos_token_id],
        )
    
    def build_prompt(self, input_output_pair: dict) -> str:
        """Build prompt using the model's chat template."""
        instruction = safety_prompt()
        messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"User Instruction: {input_output_pair['prompt']}\n\nModel Response: {input_output_pair['response']}\n\nScore:"}
    ]
    
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # Add this to prompt the model to respond
        )
        print(f"\n{'='*50}")
        print("DEBUG - Prompt format for Judge:")
        print(repr(prompt))  # repr shows special tokens
        print(f"{'='*50}\n")
        
        return prompt
    
    def evaluate_batch(self, prompts: list[dict]) -> list[dict]:
        """Evaluate all prompt-prefill combinations."""
        
        all_inputs = []
        
        for prompt in prompts:
            full_prompt = self.build_prompt(prompt)
            all_inputs.append(full_prompt)
                
        
        print(f"Running {len(all_inputs)} generations...")
        
        outputs = self.llm.generate(all_inputs, self.sampling_params)
        return outputs
        
