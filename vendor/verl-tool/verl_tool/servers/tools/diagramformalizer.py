"""
DiagramFormalizer Tool for VERL Framework
Adapted from /data/Online-GRPO-new/verl-tool/ implementation.

Formalizes geometry diagrams to CDL (Construction Description Language) format.
Model: NaughtyDog97/DiagramFormalizer (cached locally)

Usage:
    CUDA_VISIBLE_DEVICES=0 VERL_APPLY_QWEN25VL_PATCH=0 \
    python -m verl_tool.servers.serve --host localhost --port 7866 \
        --tool_type "diagramformalizer"
"""

import json
import re
import os
import logging
from typing import Tuple, Dict, Any, Optional
import base64
from io import BytesIO

import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

from .base import BaseTool, register_tool

logger = logging.getLogger(__name__)

# Local cached model path (HF cache dir — point to actual snapshot)
DIAGRAMFORMALIZER_PATH = "/path/to/VICO/tools/DiagramFormalizer/snapshots/30d103e7f3560fee83898b34677de8f9a62a3e40"
DIAGRAMFORMALIZER_MODULES = "/path/to/VICO/tools/DiagramFormalizer-modules"

DEFAULT_PROMPT = 'Based on the image, first describe what you see in the figure, then predict the construction_cdl and image_cdl and calibrate it.'


# ══════════════════════════════════════════════════════════════
#  Ray Actor
# ══════════════════════════════════════════════════════════════

@ray.remote(num_gpus=1)
class DiagramFormalizerActor:
    """
    IMPORTANT: max_concurrency must be 1. PyTorch model.generate() is NOT
    thread-safe — concurrent calls corrupt KV cache and cause CUDA errors.
    """
    def __init__(self, model_path: str = DIAGRAMFORMALIZER_PATH):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Add custom modules to path
        if os.path.exists(DIAGRAMFORMALIZER_MODULES):
            import sys
            for d in os.listdir(DIAGRAMFORMALIZER_MODULES):
                full = os.path.join(DIAGRAMFORMALIZER_MODULES, d)
                if os.path.isdir(full) and full not in sys.path:
                    sys.path.insert(0, full)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto',
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, padding_side="right",
            trust_remote_code=True,
        )
        self.logger.info(f"DiagramFormalizer loaded on {self.device}")

    def formalize_diagram(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            image_data = params.get('image_data')
            image_path = params.get('image_path')
            prompt = params.get('prompt') or DEFAULT_PROMPT
            max_new_tokens = params.get('max_new_tokens', 3500)

            results = {
                'construction_cdl': None, 'image_cdl': None,
                'description': None, 'full_response': None, 'success': False,
            }

            if image_data:
                image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
            elif image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
            else:
                results['error'] = "No valid image provided"
                return results

            response = self._generate(image, prompt, max_new_tokens)

            if response:
                results['full_response'] = response
                results['success'] = True
                cdl = self._parse_cdl(response)
                results['construction_cdl'] = cdl.get('construction_cdl')
                results['image_cdl'] = cdl.get('image_cdl')
                desc = re.search(r'^(.*?(?:construction_cdl|The construction_cdl))',
                                response, re.DOTALL)
                if desc:
                    results['description'] = desc.group(1).strip()

            return results
        except Exception as e:
            self.logger.error(f"Formalization failed: {e}", exc_info=True)
            return {"error": str(e)}

    def _generate(self, image, prompt, max_new_tokens=3500):
        text = (f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
                f'<|im_start|>user\n<image>\n{prompt}<|im_end|>\n'
                f'<|im_start|>assistant\n')

        input_ids = self._tokenizer_image_token(text, self.tokenizer, -200, 'pt')
        input_ids = input_ids.unsqueeze(0).to(self.device)

        image_tensor = self.model.process_images([image], self.model.config)
        image_tensor = image_tensor.to(dtype=self.model.dtype, device=self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=image_tensor,
                do_sample=False, temperature=None, top_p=None,
                top_k=None, num_beams=1, max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=None, use_cache=True,
            )[0]

        return self.tokenizer.decode(
            output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

    def _tokenizer_image_token(self, prompt, tokenizer, image_token_index, return_tensors=None):
        chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_sep(X, sep):
            return [e for pair in zip(X, [sep] * len(X)) for e in pair][:-1]

        input_ids = []
        offset = 0
        if chunks and chunks[0] and chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(chunks[0][0])
        for x in insert_sep(chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        return input_ids

    def _parse_cdl(self, text):
        patterns = {
            'construction_cdl': r'(?:The )?(?:calibrate )?construction_cdl(?: is)?:\n(.*?)(?=\n(?:The )?(?:calibrate )?\w+_cdl(?:\sis)?:|\nSolution is:|\Z)',
            'image_cdl': r'(?:The )?(?:calibrate )?image_cdl(?: is)?:\n(.*?)(?=\n(?:The )?(?:calibrate )?\w+_cdl(?:\sis)?:|\nSolution is:|\Z)',
        }
        results = {}
        for key, pat in patterns.items():
            for p in [pat.replace("(?:calibrate )?", "(?:calibrate )"), pat]:
                m = re.search(p, text, re.DOTALL)
                if m:
                    results[key] = m.group(1).strip()
                    break
        return results

    def health_check(self) -> str:
        if hasattr(self, 'model') and self.model is not None:
            return f"healthy: device={self.device}"
        return "unhealthy"


# ══════════════════════════════════════════════════════════════
#  Tool class
# ══════════════════════════════════════════════════════════════

@register_tool
class DiagramFormalizerTool(BaseTool):
    tool_type = "diagramformalizer"

    def __init__(self, num_workers=1, **kwargs):
        super().__init__(num_workers)
        self.logger = logging.getLogger(self.__class__.__name__)

        try:
            # max_concurrency=1: model.generate() is NOT thread-safe
            self.actor = DiagramFormalizerActor.options(
                name="DiagramFormalizerActor", get_if_exists=True,
                num_gpus=1, max_concurrency=1,
            ).remote(model_path=DIAGRAMFORMALIZER_PATH)

            health = ray.get(self.actor.health_check.remote(), timeout=120)
            self.logger.info(f"DiagramFormalizer Actor: {health}")
        except Exception as e:
            self.logger.error(f"Failed to create DiagramFormalizer Actor: {e}")
            self.actor = None

    def get_usage_inst(self):
        return (
            'DiagramFormalizer: Formalize geometry diagrams to CDL format. '
            'Use <tool_call>{"tool": "diagramformalizer", "task": "formalize"}</tool_call>'
        )

    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        matches = re.findall(r'<tool_call>(.*?)</tool_call>', action, re.DOTALL)
        if not matches:
            return {}, False
        try:
            params = json.loads(matches[0].strip())
            if params.get('tool') != 'diagramformalizer':
                return {}, False
            return {
                'task': params.get('task', 'formalize'),
                'prompt': params.get('prompt'),
                'time_limit': params.get('time_limit', 60),
                'max_new_tokens': params.get('max_new_tokens', 3500),
            }, True
        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            return {}, False

    def conduct_action(self, trajectory_id: str, action: str,
                       extra_field: Dict[str, Any]) -> Tuple[str, bool, bool]:
        parsed, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)

        if not is_valid:
            self.update_env(trajectory_id, env, action, False, extra_field, "")
            self.save_env(trajectory_id, env)
            return "", False, False

        try:
            image_path = self._get_image_path(extra_field)

            if not image_path:
                obs = "Error: No image provided for diagram formalization"
            elif self.actor is None:
                obs = "Error: DiagramFormalizer Actor not available"
            else:
                params = {
                    'image_path': image_path, 'image_data': None,
                    'prompt': parsed.get('prompt'),
                    'max_new_tokens': parsed.get('max_new_tokens', 3500),
                }
                ref = self.actor.formalize_diagram.remote(params)
                try:
                    result = ray.get(ref, timeout=parsed['time_limit'] + 10)
                    if "error" in result:
                        obs = f"Error: {result['error']}"
                    else:
                        obs = self._format_result(result)
                except ray.exceptions.GetTimeoutError:
                    obs = f"Error: Timed out ({parsed['time_limit']}s)"

            self.update_env(trajectory_id, env, parsed, is_valid, extra_field, obs)
            self.save_env(trajectory_id, env)
            return f"\n```output\n{obs}\n```\n", False, True

        except Exception as e:
            self.logger.error(f"conduct_action failed: {e}", exc_info=True)
            return f"\n```output\nError: {str(e)}\n```\n", False, True

    def _get_image_path(self, extra_field):
        if not extra_field:
            return None
        if 'images' in extra_field:
            imgs = extra_field['images']
            if imgs:
                p = imgs[0] if isinstance(imgs, list) else imgs
                if isinstance(p, str) and os.path.exists(p):
                    return p
        if 'image' in extra_field:
            p = extra_field['image']
            if isinstance(p, str) and os.path.exists(p):
                return p
        if 'image_path' in extra_field:
            p = extra_field['image_path']
            if isinstance(p, str) and os.path.exists(p):
                return p
        return None

    def _format_result(self, result):
        lines = []
        if result.get('full_response'):
            lines.append("Full Response:")
            lines.append("=" * 50)
            lines.append(result['full_response'])
            lines.append("=" * 50)
        if result.get('construction_cdl'):
            lines.append("\nConstruction CDL:")
            lines.append(result['construction_cdl'])
        if result.get('image_cdl'):
            lines.append("\nImage CDL:")
            lines.append(result['image_cdl'])
        lines.append(f"\nStatus: {'Successfully formalized' if result.get('success') else 'Failed'}")
        return "\n".join(lines)

    def update_env(self, trajectory_id, env, action, is_valid,
                   extra_field, observation, **kwargs):
        super().update_env(trajectory_id, env, action, is_valid,
                          extra_field, observation, **kwargs)
