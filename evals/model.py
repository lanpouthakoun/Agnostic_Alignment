"""
We reinstantiate our model using VLLM

"""
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from evals.detect import JudgeVLLM
class VLLMSERVE:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = None,  
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        chat_template: str = None, 
        judge_model: str = "Qwen/Qwen2.5-3B"
    ):
        print(f"Loading {model_path}...")
        
        self.model_path = model_path
        
        tokenizer_path = tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )
        
        self.judge = JudgeVLLM(judge_model = judge_model)
        if chat_template:
            self.tokenizer.chat_template = chat_template
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path,
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
    
    def build_prompt_with_prefill(self, user_message: str, prefill: str = "") -> str:
        """Build prompt using the model's chat template."""
        
        messages = [{"role": "user", "content": user_message}]
        
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not bool(prefill),
        )
        
        return prompt
    
    def evaluate_batch(self, prompts: list[str], prefill: list[str]) -> list[dict]:
        """Evaluate all prompt-prefill combinations."""
        
        all_inputs = []
        metadata = []
        
        for prompt in prompts:
            full_prompt = self.build_prompt_with_prefill(prompt, prefill)
            all_inputs.append(full_prompt)
            metadata.append({"prompt": prompt, "prefill": prefill})
        
        print(f"Running {len(all_inputs)} generations...")
        
        outputs = self.llm.generate(all_inputs, self.sampling_params)
        scores = self.judge.generate(outputs, self.sampling_params)
        results = []
        for i, output in enumerate(outputs):
            response = output.outputs[0].text.strip()
            prefill = metadata[i]["prefill"]
            
            results.append({
                "prompt": metadata[i]["prompt"],
                "prefill": prefill,
                "response": response,
                "refused": bool(scores[i]),
                "full_response": prefill + response if prefill else response,
                "tokens_generated": len(output.outputs[0].token_ids),
            })
        
        return results