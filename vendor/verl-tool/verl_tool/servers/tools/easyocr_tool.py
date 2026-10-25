"""
EasyOCR Tool for VERL Framework

Reads text/numbers from images using EasyOCR. Supports full image or cropped region.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m verl_tool.servers.serve \
        --host localhost --port 6758 --tool_type "easyocr_tool"
"""

import json
import re
import os
import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np

import torch
from PIL import Image

from .base import BaseTool, register_tool

logger = logging.getLogger(__name__)


@register_tool
class EasyOCRTool(BaseTool):
    tool_type = "easyocr_tool"

    def __init__(self, num_workers=1, **kwargs):
        super().__init__(num_workers)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.reader = None
        self._init_reader()

    def _init_reader(self):
        try:
            import easyocr
            gpu = torch.cuda.is_available()
            self.reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)
            self.logger.info(f"EasyOCR initialized (gpu={gpu})")
        except Exception as e:
            self.logger.error(f"EasyOCR init failed: {e}")

    def get_usage_inst(self):
        return (
            'EasyOCR: Read text and numbers from images. '
            'Use <tool_call>{"tool": "easyocr_tool", "bbox": [x1, y1, x2, y2]}</tool_call> '
            'where bbox is optional normalized coordinates [0-1]. '
            'Omit bbox to OCR the full image.'
        )

    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        matches = re.findall(r'<tool_call>(.*?)</tool_call>', action, re.DOTALL)
        if not matches:
            return {}, False
        try:
            params = json.loads(matches[0].strip())
            if params.get('tool') != 'easyocr_tool':
                return {}, False
            parsed = {
                'bbox': params.get('bbox', None),  # [x1,y1,x2,y2] normalized 0-1
            }
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
            image = self._get_image(extra_field, env)
            if image is None:
                obs = "Error: No image provided"
            elif self.reader is None:
                obs = "Error: EasyOCR not initialized"
            else:
                # Crop if bbox provided
                bbox = parsed_params.get('bbox')
                if bbox and len(bbox) == 4:
                    w, h = image.size
                    x1 = int(bbox[0] * w)
                    y1 = int(bbox[1] * h)
                    x2 = int(bbox[2] * w)
                    y2 = int(bbox[3] * h)
                    image = image.crop((x1, y1, x2, y2))

                # Run OCR
                img_array = np.array(image)
                results = self.reader.readtext(img_array)

                if not results:
                    obs = "No text detected in the image region."
                else:
                    lines = [f"Detected {len(results)} text regions:"]
                    for r_bbox, text, conf in results:
                        x1, y1 = int(r_bbox[0][0]), int(r_bbox[0][1])
                        x2, y2 = int(r_bbox[2][0]), int(r_bbox[2][1])
                        lines.append(f'  [{x1},{y1},{x2},{y2}] "{text}" (conf={conf:.2f})')
                    obs = "\n".join(lines)

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
        if img_str.startswith("data:image"):
            try:
                _, b64data = img_str.split(",", 1)
                return Image.open(io.BytesIO(base64.b64decode(b64data))).convert('RGB')
            except Exception:
                return None
        if os.path.exists(img_str):
            return Image.open(img_str).convert('RGB')
        return None

    def _get_image(self, extra_field, env):
        if extra_field:
            if 'images' in extra_field:
                imgs = extra_field['images']
                if imgs:
                    p = imgs[0] if isinstance(imgs, list) else imgs
                    img = self._decode_image_str(p)
                    if img is not None:
                        return img
            if 'image' in extra_field:
                img_data = extra_field['image']
                if isinstance(img_data, Image.Image):
                    return img_data
                img = self._decode_image_str(img_data)
                if img is not None:
                    return img
        if 'current_image' in env:
            return env['current_image']
        return None

    def update_env(self, trajectory_id, env, action, is_valid,
                   extra_field, observation, **kwargs):
        super().update_env(trajectory_id, env, action, is_valid,
                          extra_field, observation, **kwargs)
        if 'current_image' not in env:
            image = self._get_image(extra_field, env)
            if image:
                env['current_image'] = image
