#!/usr/bin/env python3
"""Local visual editing Planner using vLLM; prompts and tool names are configurable."""
import os, json, re
from typing import Dict, List, Optional
from PIL import Image as PILImage


class LocalPlanner:

    SYSTEM_PROMPT = """You are a visual editing planner. Choose an edit that moves
a Solver's pass rate closer to 0.5. The editor applies your instruction and must
return a valid answer for the edited image. Use <think>reasoning</think> followed
by <tool_call>{"tool":"edit","params":{"edit_instruction":"instruction"}}</tool_call>.
Consider prior feedback and adjust the difficulty gradually."""

    VALID_TOOLS = ("edit", "chart_rerender", "scene_rerender")

    def __init__(self, model_path, lora_path=None, gpu_memory_utilization=0.45,
                 max_model_len=4096, temperature=0.7, system_prompt=None, valid_tools=None):
        if system_prompt is not None:
            self.SYSTEM_PROMPT = system_prompt
        if valid_tools is not None:
            self.VALID_TOOLS = tuple(valid_tools)
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        print(f"Initializing visual reasoning Qwen3-VL Planner...")

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
            self.lora_request = LoRARequest("visual_reasoning_rewriter", 1, lora_path)
            print(f"  LoRA: {lora_path}")
        else:
            self.lora_request = None

        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(temperature=temperature, max_tokens=1024)
        self.processor = AutoProcessor.from_pretrained(model_path)

        model_label = lora_path or model_path
        print(f"visual reasoning Planner initialized (model={model_label})")

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
            prompt += "\nPrevious attempts in the edit chain:\n"
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

        prompt += "\nPlan the next edit from the current image. Adjust intensity based on prior results."
        return prompt

    def get_plan(self, image_path, question, answer, p_hat_original,
                 scene_objects=None, history=None, **kwargs):
        """
        Get an editing plan for a visual reasoning scene.

        Args:
            image_path: Path to current visual reasoning image
            question: The visual reasoning question
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

        if not isinstance(p, dict) or not p.get("edit_instruction"):
            return None

        return plan

    def _fallback_plan(self, p_hat):
        direction = "harder" if p_hat > 0.5 else "easier"
        return {"tool": "edit", "params": {"edit_instruction":
                f"Make the visual question {direction}, preserving answerability."},
                "reasoning": "Fallback based on Solver feedback"}
