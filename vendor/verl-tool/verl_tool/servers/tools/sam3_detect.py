"""
SAM3 Object Detection Tool for VERL Framework

Detects objects in images by text prompt using SAM3 (Segment Anything 3).
Returns bounding boxes, confidence scores, and optionally an annotated image.

Usage:
    conda activate sam3
    CUDA_VISIBLE_DEVICES=7 python -m verl_tool.servers.serve \
        --host localhost --port 6800 --tool_type "sam3_detect"
"""

import json
import re
import os
import logging
import threading
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import base64
from io import BytesIO

import torch
from PIL import Image, ImageDraw

from .base import BaseTool, register_tool

logger = logging.getLogger(__name__)

SAM3_MODEL_PATH = os.environ.get(
    "SAM3_MODEL_PATH",
    "/path/to/models/sam3"
)


@register_tool
class SAM3DetectTool(BaseTool):
    tool_type = "sam3_detect"

    def __init__(self, num_workers=1, model_path=None, **kwargs):
        super().__init__(num_workers)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_path = model_path or SAM3_MODEL_PATH
        self.processor = None
        self.lock = threading.Lock()
        self._load_model()

    def _load_model(self):
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            ckpt = os.path.join(self.model_path, "sam3.pt")
            bpe = os.path.join(self.model_path, "assets", "bpe_simple_vocab_16e6.txt.gz")

            self.logger.info(f"Loading SAM3 from {self.model_path}...")
            model = build_sam3_image_model(
                checkpoint_path=ckpt, load_from_HF=False, bpe_path=bpe)
            self.processor = Sam3Processor(model)
            self.logger.info("SAM3 loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load SAM3: {e}")
            self.processor = None

    def get_usage_inst(self):
        return (
            'SAM3: Detect objects by text prompt. '
            'Use <tool_call>{"tool": "sam3_detect", "query": "cube"}</tool_call> '
            'or <tool_call>{"tool": "sam3_detect", "queries": ["cube", "sphere", "cylinder"]}</tool_call>'
        )

    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        matches = re.findall(r'<tool_call>(.*?)</tool_call>', action, re.DOTALL)
        if not matches:
            return {}, False
        try:
            params = json.loads(matches[0].strip())
            tool = params.get('tool', '').lower()
            if tool not in ('sam3_detect', 'sam3', 'detect'):
                return {}, False

            # Support single query or list of queries
            queries = params.get('queries', [])
            if not queries:
                q = params.get('query', '')
                if q:
                    queries = [q]

            if not queries:
                # Default: detect common object types
                queries = ['object']

            return {'queries': queries}, True
        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            return {}, False

    def _detect(self, image: Image.Image, query: str) -> List[Dict]:
        """Run SAM3 detection for a single text query."""
        results = []
        with self.lock:
            try:
                inference_state = self.processor.set_image(image)
                state = self.processor.set_text_prompt(
                    prompt=query, state=inference_state)

                boxes = state.get("boxes")
                scores = state.get("scores")

                if boxes is not None:
                    for j in range(len(boxes)):
                        box = boxes[j]
                        score = scores[j].item() if scores is not None and j < len(scores) else 0.0
                        if isinstance(box, torch.Tensor):
                            box = box.cpu().tolist()
                        results.append({
                            "name": query,
                            "bbox": [round(b, 1) for b in box],
                            "score": round(score, 3),
                        })
            except Exception as e:
                self.logger.error(f"SAM3 detect error for '{query}': {e}")

        return results

    def _draw_bboxes(self, image: Image.Image, detections: List[Dict]) -> str:
        """Draw bboxes on image and return base64 encoded annotated image."""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        colors = ["red", "blue", "green", "yellow", "cyan",
                  "magenta", "orange", "purple"]

        for i, det in enumerate(detections):
            bbox = det["bbox"]
            color = colors[i % len(colors)]
            draw.rectangle(bbox, outline=color, width=3)
            label = f"#{i+1} {det['name']} ({det['score']:.2f})"
            draw.text((bbox[0], max(0, bbox[1] - 15)), label, fill=color)

        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _format_detections(self, detections: List[Dict]) -> str:
        """Format detection results as text."""
        if not detections:
            return "No objects detected."

        from collections import Counter
        shape_counts = Counter(d["name"] for d in detections)
        summary = ", ".join(f"{n} {s}{'s' if n > 1 else ''}"
                           for s, n in shape_counts.most_common())

        lines = [f"Detected {len(detections)} objects ({summary}):"]
        for i, det in enumerate(detections):
            lines.append(
                f"  #{i+1}: {det['name']} at bbox="
                f"[{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, "
                f"{det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}] "
                f"(confidence={det['score']:.2f})")

        return "\n".join(lines)

    def conduct_action(self, trajectory_id: str, action: str,
                       extra_field: Dict[str, Any]) -> Tuple[str, bool, bool]:
        parsed, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)

        if not is_valid:
            obs = "Error: Invalid format. Use <tool_call>{\"tool\": \"sam3_detect\", \"query\": \"...\"}</tool_call>"
            self.update_env(trajectory_id, env, action, False, extra_field, obs)
            self.save_env(trajectory_id, env)
            return f"\n```output\n{obs}\n```\n", False, False

        if self.processor is None:
            obs = "Error: SAM3 model not loaded"
            self.update_env(trajectory_id, env, parsed, is_valid, extra_field, obs)
            self.save_env(trajectory_id, env)
            return f"\n```output\n{obs}\n```\n", False, True

        try:
            # Get image
            image = self._get_image(extra_field, env)
            if image is None:
                obs = "Error: No image provided"
            else:
                # Run detection for each query
                all_detections = []
                for query in parsed['queries']:
                    dets = self._detect(image, query)
                    all_detections.extend(dets)

                obs = self._format_detections(all_detections)

                # Cache annotated image in env
                if all_detections:
                    env['last_detections'] = all_detections
                    env['annotated_b64'] = self._draw_bboxes(image, all_detections)

            self.update_env(trajectory_id, env, parsed, is_valid, extra_field, obs)
            self.save_env(trajectory_id, env)
            return f"\n```output\n{obs}\n```\n", False, True

        except Exception as e:
            self.logger.error(f"conduct_action error: {e}", exc_info=True)
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
        """Get image from extra_field."""
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
