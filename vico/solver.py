"""vLLM visual-QA Solver using the caller's CUDA device allocation."""
from pathlib import Path

from vico.metrics import PROMPTS, extract_answer, matches


class VLLMSolver:
    def __init__(self, model_path, n_rollouts=8, temperature=0.7, tp=1,
                 max_tokens=1024, max_model_len=4096, gpu_memory=0.8,
                 prompt_style="answer_tag", metric="relaxed", tolerance=0.05,
                 system_prompt=None, enforce_eager=False):
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor
        if n_rollouts < 1:
            raise ValueError("n_rollouts must be positive")
        self.llm = LLM(model=model_path, tensor_parallel_size=tp,
                       max_model_len=max_model_len, dtype="bfloat16",
                       trust_remote_code=True, gpu_memory_utilization=gpu_memory,
                       enforce_eager=enforce_eager)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.sampling = SamplingParams(n=n_rollouts, temperature=temperature, max_tokens=max_tokens)
        self.prompt_style, self.metric, self.tolerance = prompt_style, metric, tolerance
        self.system_prompt = system_prompt or PROMPTS[prompt_style]

    def evaluate(self, image, question, answer):
        from PIL import Image
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": str(Path(image))},
                                            {"type": "text", "text": question}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with Image.open(image) as im:
            pil = im.convert("RGB")
        outputs = self.llm.generate([{"prompt": prompt, "multi_modal_data": {"image": pil}}], self.sampling)
        responses = outputs[0].outputs
        if not responses:
            raise RuntimeError("Solver returned no completions")
        return sum(matches(extract_answer(r.text, self.prompt_style), answer,
                           self.metric, self.tolerance) for r in responses) / len(responses)
