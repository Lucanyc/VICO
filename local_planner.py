#!/usr/bin/env python3
"""
Qwen3-VL Planner for SCENE — local SFT'd model replacing GPT-5.

Drop-in replacement for gpt5_planner_scene.py.
Uses a merged/LoRA Qwen3-VL model via vLLM instead of OpenAI API.

Tools:
  - scene_rerender: modify scene JSON and re-render (PRIMARY)
  - blur_region: blur the image
  - add_hint: highlight target objects
  - rotate_image: rotate the 2D image
"""
import os, json, re
from typing import Dict, List, Optional
from PIL import Image as PILImage


class LocalPlanner:

    SYSTEM_PROMPT = """You are an expert SCENE scene editing planner.

Your task: choose one editing action to move a Solver model's pass rate (p̂) closer to 0.5.

Available tools:
- scene_rerender(edit_instruction): modify the scene's object list (add/remove/change objects) and re-render with Blender. This is the PRIMARY tool for SCENE.
- blur_region(sigma): blur the image to make object identification harder. Parameter sigma: float, 3.0-50.0.
- add_hint(level): add visual highlights to make target objects easier to find. Parameter level: integer, 1-5.
- rotate_image(angle): rotate the image to make spatial reasoning harder. Parameter angle: float, 5.0-180.0.

When using scene_rerender:
- ALWAYS preserve the correct answer
- To make harder: add visual distractors (similar color/shape but different type)
- To make easier: remove distractors, simplify the scene
- Be specific in edit instructions

First think step-by-step about what edit to apply, then output the tool call.
Your response format:
<think>
your reasoning
</think>
<tool_call>
{"tool": "scene_rerender", "params": {"edit_instruction": "your specific edit instruction"}}
</tool_call>"""

    VALID_TOOLS = ("scene_rerender", "blur_region", "add_hint", "rotate_image")

    def __init__(self, model_path, lora_path=None, gpu_memory_utilization=0.45,
                 max_model_len=4096, temperature=0.7):
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        print(f"Initializing SCENE Qwen3-VL Planner...")

        llm_kwargs = {
            "model": model_path,
            "trust_remote_code": True,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "disable_log_stats": True,
        }

        if lora_path:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = 64
            from vllm.lora.request import LoRARequest
            self.lora_request = LoRARequest("scene_rewriter", 1, lora_path)
            print(f"  LoRA: {lora_path}")
        else:
            self.lora_request = None

        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(temperature=temperature, max_tokens=1024)
        self.processor = AutoProcessor.from_pretrained(model_path)

        model_label = lora_path or model_path
        print(f"SCENE Planner initialized (model={model_label})")

    def _format_scene_objects(self, scene_objects):
        """Format scene objects into a readable string."""
        if not scene_objects:
            return "Scene objects not available."
        lines = [f"Scene objects ({len(scene_objects)} total):"]
        for i, obj in enumerate(scene_objects):
            color = obj.get("color", "?")
            material = obj.get("material", "?")
            size = obj.get("size", "?")
            shape = obj.get("shape", "?")
            coords = obj.get("3d_coords", [0, 0, 0])
            lines.append(f"  [{i}] {size} {color} {material} {shape} "
                         f"at ({coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f})")
        return "\n".join(lines)

    def _build_user_prompt(self, question, answer, p_hat, scene_objects, history):
        prompt = f"Question: {question}\nAnswer: {answer}\n"
        prompt += f"Current Solver pass rate (p̂): {p_hat:.3f}\n\n"
        prompt += self._format_scene_objects(scene_objects) + "\n"

        if history:
            prompt += "\nPrevious attempts (each started from the ORIGINAL image above):\n"
            for i, h in enumerate(history, 1):
                tool = h.get("tool", "?")
                if tool == "scene_rerender":
                    instr = h.get("params", {}).get("edit_instruction",
                            h.get("params", {}).get("direction", ""))[:80]
                    ps = f'edit_instruction="{instr}..."'
                else:
                    ps = ", ".join(f"{k}={v}" for k, v in h.get("params", {}).items())
                p_val = h.get("p_hat", "?")
                p_str = f"{p_val:.3f}" if isinstance(p_val, (int, float)) else str(p_val)
                err = h.get("error", "")
                err_str = f" [FAILED: {err[:40]}]" if err else ""
                prompt += f"  Attempt {i}: {tool}({ps}) → p̂={p_str}{err_str}\n"

            phats = [h["p_hat"] for h in history if isinstance(h.get("p_hat"), (int, float))]
            if phats:
                closest = min(phats, key=lambda x: abs(x - 0.5))
                too_hard = [p for p in phats if p < 0.35]
                too_easy = [p for p in phats if p > 0.65]
                prompt += f"\nCalibration: {len(history)} attempts, closest p̂={closest:.3f}."
                if too_hard:
                    prompt += f" {len(too_hard)} were too aggressive — try FEWER/smaller modifications."
                if too_easy:
                    prompt += f" {len(too_easy)} were too mild — try MORE/larger modifications."
                prompt += "\n"
        else:
            prompt += "\nNo edits applied yet. This is the original image.\n"

        prompt += "\nPlan a NEW edit from the original image. Adjust intensity based on prior results."
        return prompt

    def get_plan(self, image_path, question, answer, p_hat_original,
                 scene_objects=None, history=None, **kwargs):
        """
        Get an editing plan for a SCENE scene.

        Args:
            image_path: Path to current SCENE image
            question: The SCENE question
            answer: The correct answer
            p_hat_original: Current solver pass rate
            scene_objects: List of objects in the scene
            history: Previous edit history

        Returns:
            dict with keys: tool, params, reasoning
        """
        scene_objects = scene_objects or []
        history = history or []
        user_prompt = self._build_user_prompt(
            question, answer, p_hat_original, scene_objects, history)

        # Build chat messages
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": self.SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_prompt},
            ]},
        ]

        prompt = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)

        try:
            pil = PILImage.open(image_path).convert("RGB")
            # Resize if too large
            max_pixels = 4096 * 4096
            w, h = pil.size
            if w * h > max_pixels:
                ratio = (max_pixels / (w * h)) ** 0.5
                pil = pil.resize((int(w * ratio), int(h * ratio)), PILImage.LANCZOS)
        except Exception as e:
            print(f"    [Planner] Cannot open image: {e}")
            return self._fallback_plan(p_hat_original)

        try:
            generate_kwargs = {
                "sampling_params": self.sampling_params,
            }
            if self.lora_request:
                generate_kwargs["lora_request"] = self.lora_request

            outs = self.llm.generate(
                [{"prompt": prompt, "multi_modal_data": {"image": pil}}],
                **generate_kwargs)

            raw = outs[0].outputs[0].text.strip()
            plan = self._parse_plan(raw)
            if plan:
                return plan
            print(f"    [Planner] Parse failed, raw: {raw[:200]}")
        except Exception as e:
            print(f"    [Planner] Generation error: {e}")

        return self._fallback_plan(p_hat_original)

    def _parse_plan(self, raw):
        """Parse model output into a structured plan."""
        # Extract reasoning from <think> tags
        reasoning = ""
        think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
        if think_match:
            reasoning = think_match.group(1).strip()

        # Extract tool call from <tool_call> tags
        tc_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', raw, re.DOTALL)
        if tc_match:
            try:
                plan = json.loads(tc_match.group(1).strip())
                plan.setdefault("reasoning", reasoning)
                return self._validate_plan(plan)
            except json.JSONDecodeError:
                pass

        # Fallback: try to find JSON anywhere in output
        json_match = re.search(r'\{[^{}]*"tool"\s*:\s*"[^"]+?"[^{}]*\}', raw)
        if json_match:
            try:
                plan = json.loads(json_match.group(0))
                plan.setdefault("reasoning", reasoning or "(extracted from fragments)")
                return self._validate_plan(plan)
            except json.JSONDecodeError:
                pass

        return None

    def _validate_plan(self, plan):
        """Validate and clamp plan parameters."""
        if not isinstance(plan, dict):
            return None
        if plan.get("tool") not in self.VALID_TOOLS:
            return None

        plan.setdefault("params", {})
        plan.setdefault("reasoning", "")

        t, p = plan["tool"], plan["params"]

        if t == "scene_rerender":
            if "edit_instruction" not in p or not p["edit_instruction"]:
                return None
        elif t == "blur_region":
            p["sigma"] = max(3.0, min(50.0, float(p.get("sigma", 10.0))))
        elif t == "add_hint":
            p["level"] = max(1, min(5, int(p.get("level", 3))))
        elif t == "rotate_image":
            p["angle"] = max(5.0, min(180.0, float(p.get("angle", 30.0))))

        return plan

    def _fallback_plan(self, p_hat):
        """Fallback plan when model output is unparseable."""
        if p_hat > 0.65:
            return {"tool": "scene_rerender",
                    "params": {"edit_instruction": "Add 2 distractor objects with similar colors to existing objects"},
                    "reasoning": "Fallback: add distractors to increase difficulty"}
        elif p_hat < 0.35:
            return {"tool": "scene_rerender",
                    "params": {"edit_instruction": "Remove 2 objects that are not relevant to the question to simplify the scene"},
                    "reasoning": "Fallback: simplify scene to decrease difficulty"}
        return {"tool": "blur_region", "params": {"sigma": 5.0},
                "reasoning": "Fallback: minimal adjustment"}


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/path/to/VICO/scene_round0/rewriter_merged_step100"

    planner = LocalPlanner(model_path=model_path, gpu_memory_utilization=0.5)

    test_image = "/path/to/dataset/Scene-Math/images/train/SCENE_train_000000.png"
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        sys.exit(1)

    test_objects = [
        {"shape": "cube", "size": "large", "material": "rubber", "color": "blue",
         "3d_coords": [-1.37, 2.08, 0.7]},
        {"shape": "cylinder", "size": "large", "material": "metal", "color": "brown",
         "3d_coords": [-0.25, -2.31, 0.7]},
        {"shape": "cube", "size": "small", "material": "rubber", "color": "gray",
         "3d_coords": [1.02, -1.94, 0.35]},
        {"shape": "sphere", "size": "large", "material": "metal", "color": "brown",
         "3d_coords": [0.44, 3.0, 0.7]},
        {"shape": "cube", "size": "large", "material": "metal", "color": "cyan",
         "3d_coords": [2.54, 0.58, 0.7]},
        {"shape": "cube", "size": "large", "material": "rubber", "color": "red",
         "3d_coords": [-2.41, -0.89, 0.7]},
    ]

    sep = "=" * 60

    print(sep)
    print("Test 1: p_hat=1.0 (too easy)")
    print(sep)
    plan = planner.get_plan(
        image_path=test_image,
        question="How many cubes are there?",
        answer="4",
        p_hat_original=1.0,
        scene_objects=test_objects,
    )
    print(f"  Tool: {plan['tool']}")
    print(f"  Params: {json.dumps(plan['params'])}")
    print(f"  Reasoning: {plan['reasoning'][:120]}")

    print()
    print(sep)
    print("Test 2: p_hat=0.0 (too hard)")
    print(sep)
    plan = planner.get_plan(
        image_path=test_image,
        question="How many cubes are there?",
        answer="4",
        p_hat_original=0.0,
        scene_objects=test_objects,
    )
    print(f"  Tool: {plan['tool']}")
    print(f"  Params: {json.dumps(plan['params'])}")
    print(f"  Reasoning: {plan['reasoning'][:120]}")

    print()
    print(sep)
    print("Test 3: p_hat=0.875 with history")
    print(sep)
    plan = planner.get_plan(
        image_path=test_image,
        question="How many cubes are there?",
        answer="4",
        p_hat_original=0.875,
        scene_objects=test_objects,
        history=[
            {"tool": "scene_rerender",
             "params": {"edit_instruction": "Add a small red metal sphere as distractor"},
             "p_hat": 0.875,
             "reasoning": "Added distractor but p_hat still high"},
        ],
    )
    print(f"  Tool: {plan['tool']}")
    print(f"  Params: {json.dumps(plan['params'])}")
    print(f"  Reasoning: {plan['reasoning'][:120]}")