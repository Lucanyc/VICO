"""
Calculator Tool for VERL Framework

Executes Python math expressions safely. For arithmetic, statistics, etc.

Usage:
    python -m verl_tool.servers.serve --host localhost --port 6700 --tool_type "calculator"
"""

import json
import re
import math
import logging
from typing import Tuple, Dict, Any

from .base import BaseTool, register_tool

logger = logging.getLogger(__name__)

# Safe builtins for eval
SAFE_GLOBALS = {
    "__builtins__": {},
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "len": len, "sorted": sorted,
    "int": int, "float": float, "str": str,
    "pow": pow, "divmod": divmod,
    "math": math,
    "pi": math.pi, "e": math.e,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "ceil": math.ceil, "floor": math.floor,
    "gcd": math.gcd,
}


@register_tool
class CalculatorTool(BaseTool):
    tool_type = "calculator"

    def __init__(self, num_workers=1, **kwargs):
        super().__init__(num_workers)

    def get_usage_inst(self):
        return (
            'Calculator: Execute Python math expressions. '
            'Use <tool_call>{"tool": "calculator", "expression": "0.5 * 6 * 4"}</tool_call>'
        )

    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        matches = re.findall(r'<tool_call>(.*?)</tool_call>', action, re.DOTALL)
        if not matches:
            return {}, False
        try:
            params = json.loads(matches[0].strip())
            if params.get('tool') != 'calculator':
                return {}, False
            expr = params.get('expression', '')
            if not expr:
                return {}, False
            return {'expression': expr}, True
        except:
            return {}, False

    def conduct_action(self, trajectory_id: str, action: str,
                       extra_field: Dict[str, Any]) -> Tuple[str, bool, bool]:
        parsed, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)

        if not is_valid:
            obs = "Error: Invalid format. Use {\"tool\": \"calculator\", \"expression\": \"...\"}"
            self.update_env(trajectory_id, env, action, False, extra_field, obs)
            self.save_env(trajectory_id, env)
            return f"\n```output\n{obs}\n```\n", False, False

        expr = parsed['expression']
        try:
            # Security: limit expression length
            if len(expr) > 500:
                obs = "Error: Expression too long (max 500 chars)"
            else:
                result = eval(expr, SAFE_GLOBALS, {})
                obs = str(result)
        except Exception as e:
            obs = f"Error: {str(e)}"

        self.update_env(trajectory_id, env, parsed, is_valid, extra_field, obs)
        self.save_env(trajectory_id, env)
        return f"\n```output\n{obs}\n```\n", False, True
