"""
ChartMoE Tool for VERL Framework
Adapted from 同事's implementation at /data/Online-GRPO-new/verl-tool/

Extracts structured table data from chart/plot images using ChartMoE model.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m verl_tool.servers.serve \
        --host localhost --port 6658 --tool_type "chartmoe"
"""

import sys

# ChartMoE custom code path
CHARTMOE_MODEL_PATH = "/path/to/VICO/tools/cache/models--IDEA-FinAI--chartmoe/snapshots/951f46f96e307fa91c68ba9f318b30e6f9ce4405"
sys.path.insert(0, CHARTMOE_MODEL_PATH)

import json
import re
import os
import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np

import ray
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseTool, register_tool

os.environ["HF_TORCH_LOAD_DISABLE_SAFE_CHECK"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "1"

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  Ray Actor — model holder
# ══════════════════════════════════════════════════════════════

@ray.remote(num_gpus=1 if torch.cuda.is_available() else 0)
class ChartMoEActor:
    def __init__(self, model_path: str = CHARTMOE_MODEL_PATH):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path

        self.model, self.tokenizer = self._load_model(model_path)

        self.image_size = 490
        self.preprocess = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size),
                             interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])

        if self.model:
            self.logger.info("ChartMoE Actor initialized successfully.")

    def _load_model(self, model_path):
        try:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            device_map = "auto" if torch.cuda.is_available() else "cpu"

            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True,
                torch_dtype=dtype, device_map=device_map,
                local_files_only=False,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True,
                local_files_only=False,
            )
            model_size = sum(p.numel() for p in model.parameters()) / 1e9
            self.logger.info(f"ChartMoE loaded: {model_size:.2f}B params")
            return model, tokenizer
        except Exception as e:
            self.logger.error(f"Failed to load ChartMoE: {e}")
            return None, None

    def inference(self, image_data: np.ndarray, params: Dict[str, Any]) -> str:
        if not self.model:
            return "Error: ChartMoE model not available."
        try:
            image = Image.fromarray(image_data.astype('uint8'), 'RGB')

            task = params.get('task', 'to_table')
            prompt = params.get('prompt', '')
            question = params.get('question', '')

            task_prompts = {
                'to_table': 'Convert this chart to a table format with clear rows and columns.',
                'describe': 'Describe this chart in detail.',
                'extract_data': 'Extract all numerical data values and labels from this chart.',
                'summarize': 'Provide a brief summary of what this chart shows.',
                'analyze': 'Analyze this chart and provide key insights.',
                'answer': '',
            }

            if task == 'answer':
                query = question if question else prompt
                if not query:
                    query = "What does this chart show?"
            elif prompt:
                query = prompt
            else:
                query = task_prompts.get(task, "Analyze this chart.")

            formatted_query = f"<ImageHere>{query}"

            image_tensor = self.preprocess(image).unsqueeze(0).to(
                next(self.model.parameters()).device)
            if torch.cuda.is_available():
                image_tensor = image_tensor.half()

            response = self.model.chat(
                tokenizer=self.tokenizer, query=formatted_query,
                image=image_tensor, history=[],
                max_new_tokens=800, do_sample=False, use_cache=True,
            )
            result = response[0] if isinstance(response, tuple) else response
            return result
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            return f"Error: {str(e)}"

    def health_check(self) -> str:
        if self.model is not None:
            return f"healthy: model loaded from {self.model_path}"
        return "unhealthy: model not loaded"


# ══════════════════════════════════════════════════════════════
#  Tool class
# ══════════════════════════════════════════════════════════════

@register_tool
class ChartMoETool(BaseTool):
    tool_type = "chartmoe"

    def __init__(self, num_workers=1, model_path=None, **kwargs):
        super().__init__(num_workers)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_path = model_path or os.environ.get(
            'CHARTMOE_MODEL_PATH', CHARTMOE_MODEL_PATH)

        self.logger.info(f"ChartMoE model path: {self.model_path}")

        try:
            num_gpus = 1 if torch.cuda.is_available() else 0
            self.actor = ChartMoEActor.options(
                name="ChartMoEActor", get_if_exists=True,
                num_gpus=num_gpus, max_concurrency=3,
            ).remote(model_path=self.model_path)

            health = ray.get(self.actor.health_check.remote(), timeout=700)
            self.logger.info(f"ChartMoE Actor: {health}")
        except Exception as e:
            self.logger.error(f"Failed to create ChartMoE Actor: {e}")
            self.actor = None

    def get_usage_inst(self):
        return (
            'ChartMoE: Chart table extraction tool. '
            'Use <tool_call>{"tool": "chartmoe", "task": "to_table|describe|extract_data|answer", '
            '"prompt": "optional", "question": "for Q&A"}</tool_call>'
        )

    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        matches = re.findall(r'<tool_call>(.*?)</tool_call>', action, re.DOTALL)
        if not matches:
            return {}, False
        try:
            params = json.loads(matches[0].strip())
            if params.get('tool') != 'chartmoe':
                return {}, False

            inner = params.get('parameters', params)
            parsed = {
                'task': inner.get('task', 'to_table'),
                'prompt': inner.get('prompt', ''),
                'question': inner.get('question', ''),
                'output_format': inner.get('output_format', 'detailed'),
            }
            if parsed['question'] and not parsed['prompt']:
                parsed['prompt'] = parsed['question']

            valid_tasks = ['to_table', 'describe', 'extract_data',
                          'summarize', 'analyze', 'answer']
            if parsed['task'] not in valid_tasks:
                parsed['task'] = 'analyze'

            return parsed, True
        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            return {}, False

    def conduct_action(self, trajectory_id: str, action: str,
                       extra_field: Dict[str, Any]) -> Tuple[str, bool, bool]:
        parsed_params, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)

        if not is_valid:
            obs = "Error: Invalid format. Use <tool_call>{...}</tool_call>"
            self.update_env(trajectory_id, env, action, False, extra_field, obs)
            self.save_env(trajectory_id, env)
            return f"\n```output\n{obs}\n```\n", False, False

        try:
            image = self._get_image(trajectory_id, extra_field, env)

            if image is None:
                obs = "Error: No image provided for chart analysis"
            elif self.actor is None:
                obs = "Error: ChartMoE Actor not available"
            else:
                image_array = np.array(image.convert('RGB'))
                result_ref = self.actor.inference.remote(image_array, parsed_params)
                try:
                    result = ray.get(result_ref, timeout=900)
                    obs = self._format_output(result, parsed_params['task'])
                except ray.exceptions.GetTimeoutError:
                    obs = "Error: ChartMoE inference timed out"

            self.update_env(trajectory_id, env, parsed_params,
                           is_valid, extra_field, obs)
            self.save_env(trajectory_id, env)
            return f"\n```output\n{obs}\n```\n", False, True

        except Exception as e:
            self.logger.error(f"conduct_action failed: {e}", exc_info=True)
            obs = f"Error: {str(e)}"
            return f"\n```output\n{obs}\n```\n", False, True

    @staticmethod
    def _decode_image_str(img_str):
        """Decode image from file path or base64 data URL."""
        import base64, io
        if not isinstance(img_str, str):
            return None
        # base64 data URL from verl-tool agent loop
        if img_str.startswith("data:image"):
            # "data:image/jpeg;base64,/9j/4AAQ..."
            try:
                header, b64data = img_str.split(",", 1)
                img_bytes = base64.b64decode(b64data)
                return Image.open(io.BytesIO(img_bytes)).convert('RGB')
            except Exception:
                return None
        # file path
        if os.path.exists(img_str):
            return Image.open(img_str).convert('RGB')
        return None

    def _get_image(self, trajectory_id, extra_field, env):
        """Get image from extra_field (verl-tool format)."""
        if extra_field:
            # verl-tool format: images list (may be file paths or base64 data URLs)
            if 'images' in extra_field:
                images_list = extra_field['images']
                if images_list:
                    img_data = images_list[0] if isinstance(images_list, list) else images_list
                    img = self._decode_image_str(img_data)
                    if img is not None:
                        return img

            # Fallback: image field
            if 'image' in extra_field and extra_field['image']:
                img_data = extra_field['image']
                if isinstance(img_data, Image.Image):
                    return img_data
                img = self._decode_image_str(img_data)
                if img is not None:
                    return img

            # Fallback: image_path field
            if 'image_path' in extra_field and extra_field['image_path']:
                path_info = extra_field['image_path']
                if isinstance(path_info, list) and path_info:
                    p = path_info[0].get("image_url") if isinstance(path_info[0], dict) else path_info[0]
                    img = self._decode_image_str(p) if p else None
                    if img is not None:
                        return img
                elif isinstance(path_info, str):
                    img = self._decode_image_str(path_info)
                    if img is not None:
                        return img

        # Cached in env
        if 'current_image' in env:
            return env['current_image']
        return None

    def _format_output(self, result, task):
        if result.startswith("Error:"):
            return result
        if task == 'to_table' and '|' not in result:
            lines = result.strip().split('\n')
            if len(lines) > 1:
                formatted = []
                for line in lines:
                    if line.strip():
                        if '\t' in line:
                            cells = line.split('\t')
                        elif ',' in line:
                            cells = line.split(',')
                        else:
                            formatted.append(f"| {line.strip()} |")
                            continue
                        formatted.append('| ' + ' | '.join(c.strip() for c in cells) + ' |')
                if len(formatted) > 1:
                    n_cols = formatted[0].count('|') - 1
                    if n_cols > 1:
                        formatted.insert(1, '|' + '---|' * n_cols)
                return '\n'.join(formatted)
        return result

    def update_env(self, trajectory_id, env, action, is_valid,
                   extra_field, observation, **kwargs):
        super().update_env(trajectory_id, env, action, is_valid,
                          extra_field, observation, **kwargs)
        if 'current_image' not in env:
            image = self._get_image(trajectory_id, extra_field, env)
            if image:
                env['current_image'] = image
