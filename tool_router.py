"""
Tool Router for Co-evolve GRPO Training

Routes tool calls to the correct tool server based on <tool_call> content.
Each tool runs on its own port/env. The router dispatches and collects results.

Tools:
  - chartmoe:           Chart → table extraction (port 6658, chartmoe env)
  - easyocr_tool:       OCR text detection (port 6758, easyocr env)
  - calculator:          Python math eval (port 6700, any env)
  - diagramformalizer:  Geometry → CDL (port 7866, verl-tool env)
  - sam3:               Object detection (port 6800, sam3 env)
  - pixel_reasoner:     Crop/zoom image (port 6900, verl-tool env)
  - finish:             End trajectory (routed to default)

Usage:
    python tool_router.py  # listens on port 5556
"""

import asyncio
import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
import re

app = FastAPI(title="Co-evolve Tool Router")

# ═══════════════════════════════════════════════════════════════
#  Tool → Port mapping
# ═══════════════════════════════════════════════════════════════

# Tool backends: each tool maps to a LIST of "host:port" endpoints.
# Multiple ChartMoE instances across GPUs/nodes = higher throughput.
# Router round-robins requests across available backends.
#
# Format: "host:port" or just "port" (defaults to localhost)
TOOL_PORTS = {
    # Primary names → list of endpoints
    "chartmoe":          ["localhost:6658", "localhost:6659", "localhost:6660", "localhost:6661"],
    "easyocr_tool":      ["localhost:6758"],
    "calculator":        ["localhost:6700"],
    "diagramformalizer": ["localhost:7866"],
    "pixel_reasoner":    ["localhost:6900"],
    "sam3_detect":       ["localhost:6800"],
    "sam3":              ["localhost:6800"],
    # Aliases
    "easyocr":           ["localhost:6758"],
    "ocr":               ["localhost:6758"],
    "extract_table":     ["localhost:6658", "localhost:6659", "localhost:6660", "localhost:6661"],
    "calculate":         ["localhost:6700"],
    "crop_zoom":         ["localhost:6900"],
    "detect":            ["localhost:6800"],
    "finish":            ["localhost:6700"],
}

DEFAULT_ENDPOINT = "localhost:6700"

# Round-robin counters per tool
import itertools
_tool_rr_counters = {tool: itertools.cycle(range(len(eps))) for tool, eps in TOOL_PORTS.items()}

def get_endpoint_for_tool(tool: str) -> str:
    """Get next endpoint (host:port) for tool via round-robin."""
    endpoints = TOOL_PORTS.get(tool, [DEFAULT_ENDPOINT])
    if len(endpoints) == 1:
        return endpoints[0]
    idx = next(_tool_rr_counters.get(tool, itertools.cycle([0])))
    return endpoints[idx]

def endpoint_to_url(endpoint: str) -> str:
    """Convert 'host:port' to 'http://host:port/get_observation'."""
    if not endpoint.startswith("http"):
        endpoint = f"http://{endpoint}"
    return f"{endpoint}/get_observation"


# ═══════════════════════════════════════════════════════════════
#  Tool name extraction
# ═══════════════════════════════════════════════════════════════

def normalize_tool_call(action: str) -> str:
    """
    Normalize model's tool_call to match what each tool server expects.
    Model learns WHAT tool to call, router handles HOW to call it.
    """
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', action, re.DOTALL)
    if not matches:
        return action

    try:
        params = json.loads(matches[0].strip())
    except:
        return action

    tool = params.get('tool', '').lower()

    # Calculator: model sends {"tool":"calculator","task":"compute"} or {"tool":"calculator","expression":"..."}
    # Server expects: {"tool":"calculator","expression":"..."}
    if tool in ('calculator', 'calculate'):
        expr = params.get('expression', '')
        if not expr:
            # Try to extract expression from task field or other fields
            task = params.get('task', '')
            if task and task != 'compute':
                expr = task  # model might put the expression in task
            else:
                # Look for any numeric/math content in other fields
                for k, v in params.items():
                    if k not in ('tool', 'task') and isinstance(v, str) and v:
                        expr = v
                        break
        if expr:
            normalized = json.dumps({"tool": "calculator", "expression": expr})
            return action.replace(matches[0], normalized)

    # SAM3: model sends {"tool":"sam3_detect","task":"detect_objects"}
    # Server expects: {"tool":"sam3_detect","queries":["cube","sphere","cylinder"]}
    if tool in ('sam3_detect', 'sam3', 'detect'):
        queries = params.get('queries', [])
        if not queries:
            # Default queries for SCENE-style scenes
            queries = ["cube", "sphere", "cylinder"]
        normalized = json.dumps({"tool": "sam3_detect", "queries": queries})
        return action.replace(matches[0], normalized)

    # ChartMoE: model sends various formats, server expects {"tool":"chartmoe","task":"to_table"}
    if tool in ('chartmoe', 'extract_table'):
        normalized = json.dumps({"tool": "chartmoe", "task": params.get("task", "to_table")})
        return action.replace(matches[0], normalized)

    # EasyOCR: model sends various formats, server just needs tool name
    if tool in ('easyocr_tool', 'easyocr', 'ocr'):
        normalized = json.dumps({"tool": "easyocr_tool"})
        return action.replace(matches[0], normalized)

    # DiagramFormalizer
    if tool in ('diagramformalizer',):
        normalized = json.dumps({"tool": "diagramformalizer", "task": params.get("task", "formalize")})
        return action.replace(matches[0], normalized)

    return action


def extract_tool_from_action(action: str) -> str:
    """Extract tool name from <tool_call>{"tool": "xxx", ...}</tool_call>"""
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', action, re.DOTALL)
    if matches:
        try:
            params = json.loads(matches[0].strip())
            tool = params.get('tool', '').lower()
            if tool:
                return tool
        except:
            pass

    # Keyword fallback
    action_lower = action.lower()
    for tool_name in TOOL_PORTS:
        if tool_name in action_lower:
            return tool_name

    return None


# ═══════════════════════════════════════════════════════════════
#  Output formatting
# ═══════════════════════════════════════════════════════════════

def format_tool_output(tool_name: str, raw_obs) -> str:
    """Format tool output for the model to consume."""
    if isinstance(raw_obs, str) and raw_obs.startswith("Error"):
        return raw_obs

    prefixes = {
        "chartmoe": "ChartMoE Table Extraction:\n",
        "easyocr_tool": "EasyOCR Detection:\n",
        "easyocr": "EasyOCR Detection:\n",
        "ocr": "EasyOCR Detection:\n",
        "calculator": "Calculation Result:\n",
        "calculate": "Calculation Result:\n",
        "diagramformalizer": "Diagram Formalization:\n",
        "sam3": "Object Detection:\n",
        "detect": "Object Detection:\n",
        "pixel_reasoner": "Cropped Region:\n",
        "crop_zoom": "Cropped Region:\n",
    }

    prefix = prefixes.get(tool_name, "Tool Output:\n")

    if isinstance(raw_obs, dict):
        # Try common output fields
        for key in ['full_response', 'output', 'processed_output', 'result', 'table_data']:
            if key in raw_obs and raw_obs[key]:
                return prefix + str(raw_obs[key])
        # EasyOCR detection format
        if 'detections' in raw_obs and isinstance(raw_obs['detections'], list):
            items = []
            for det in raw_obs['detections']:
                text = det.get('text', '')
                bbox = det.get('bbox', [])
                if bbox and len(bbox) >= 4:
                    items.append(f'{text} [{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]')
                else:
                    items.append(text)
            if items:
                return prefix + " | ".join(items)
            return prefix + "No text detected"
        return prefix + json.dumps(raw_obs, ensure_ascii=False, default=str)

    return prefix + str(raw_obs)


# ═══════════════════════════════════════════════════════════════
#  Router endpoint
# ═══════════════════════════════════════════════════════════════

@app.post("/get_observation")
async def route_observation(request: Request):
    data = await request.json()

    actions = data.get("actions", [])
    n = len(actions)

    # Group by target port
    tool_groups = {}  # port → {indices, data}

    for i, action in enumerate(actions):
        is_finish = data.get("finish", [False] * n)[i]

        if is_finish:
            tool = "finish"
        else:
            tool = extract_tool_from_action(action)

        endpoint = get_endpoint_for_tool(tool) if tool else DEFAULT_ENDPOINT

        if endpoint not in tool_groups:
            tool_groups[endpoint] = {
                "indices": [],
                "data": {
                    "trajectory_ids": [],
                    "actions": [],
                    "finish": [],
                }
            }

        g = tool_groups[endpoint]
        g["indices"].append(i)
        g["data"]["trajectory_ids"].append(data["trajectory_ids"][i])
        # Normalize tool call format before dispatching
        g["data"]["actions"].append(normalize_tool_call(action))
        g["data"]["finish"].append(is_finish)

        # Pass through extra fields
        if "extra_fields" in data:
            g["data"].setdefault("extra_fields", []).append(data["extra_fields"][i])
        if "is_last_step" in data:
            g["data"].setdefault("is_last_step", []).append(data["is_last_step"][i])

    # Log routing
    for endpoint, g in tool_groups.items():
        tools_at_ep = set()
        for idx in g["indices"]:
            t = extract_tool_from_action(actions[idx])
            if t: tools_at_ep.add(t)
        print(f"  → {endpoint}: {len(g['indices'])} requests ({tools_at_ep or {'finish'}})")

    # Dispatch to tool servers concurrently
    results = {
        "observations": [None] * n,
        "dones": [False] * n,
        "valids": [False] * n,
    }

    async with aiohttp.ClientSession() as session:
        tasks = []
        for endpoint, g in tool_groups.items():
            url = endpoint_to_url(endpoint)
            tasks.append((endpoint, g["indices"], session.post(url, json=g["data"])))

        for endpoint, indices, task in tasks:
            try:
                async with task as response:
                    if response.status == 200:
                        resp = await response.json()
                        for j, orig_idx in enumerate(indices):
                            raw_obs = resp["observations"][j]
                            tool = extract_tool_from_action(actions[orig_idx])
                            results["observations"][orig_idx] = format_tool_output(tool, raw_obs)
                            results["dones"][orig_idx] = resp["dones"][j]
                            results["valids"][orig_idx] = resp["valids"][j]
                    else:
                        err = await response.text()
                        print(f"  ERROR {endpoint}: {response.status} - {err[:100]}")
                        for idx in indices:
                            results["observations"][idx] = f"Error: port {port} returned {response.status}"
                            results["dones"][idx] = True
                            results["valids"][idx] = False
            except Exception as e:
                print(f"  ERROR {endpoint}: {e}")
                for idx in indices:
                    results["observations"][idx] = f"Error: {str(e)[:100]}"
                    results["dones"][idx] = True
                    results["valids"][idx] = False

    return JSONResponse(content=results)


# ═══════════════════════════════════════════════════════════════
#  Health check — router + all tool backends
# ═══════════════════════════════════════════════════════════════

async def check_tool_status():
    """Check all tool ports for health."""
    # Deduplicate ports (some tools share ports via aliases)
    unique_ports = {}
    for tool, ports in TOOL_PORTS.items():
        for port in ports:
            if port not in unique_ports:
                unique_ports[port] = []
            if tool not in unique_ports[port]:
                unique_ports[port].append(tool)

    status = {}
    async with aiohttp.ClientSession() as session:
        for endpoint, tools in unique_ports.items():
            # endpoint is already "host:port" format
            tool_label = f"{tools[0]}@{endpoint}"
            health_url = f"http://{endpoint}/health"
            try:
                async with session.get(
                    health_url,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    status[tool_label] = {
                        "endpoint": endpoint,
                        "aliases": tools,
                        "status": "online",
                        "code": resp.status,
                    }
            except aiohttp.ClientConnectorError:
                status[tool_label] = {
                    "endpoint": endpoint,
                    "aliases": tools,
                    "status": "offline (connection refused)",
                }
            except asyncio.TimeoutError:
                status[tool_label] = {
                    "endpoint": endpoint,
                    "aliases": tools,
                    "status": "timeout (5s)",
                }
            except Exception as e:
                status[tool_label] = {
                    "endpoint": endpoint,
                    "aliases": tools,
                    "status": f"error: {str(e)}",
                }
    return status


@app.on_event("startup")
async def startup_check():
    """Check all tool backends on startup."""
    print("\n" + "=" * 60)
    print("Tool Health Check (startup)")
    print("=" * 60)

    status = await check_tool_status()
    online = sum(1 for v in status.values() if v["status"] == "online")
    total = len(status)

    for tool, info in status.items():
        icon = "[OK]" if info["status"] == "online" else "[--]"
        print(f"  {icon} {tool:30s} {info['endpoint']}  {info['status']}")

    print(f"\n  {online}/{total} tool backends online")
    if online < total:
        offline = [t for t, v in status.items() if v["status"] != "online"]
        print(f"  Offline: {', '.join(offline)}")
    print("=" * 60 + "\n")


@app.get("/check_tools")
async def check_tools():
    """HTTP endpoint to check all tool backends."""
    status = await check_tool_status()
    online = sum(1 for v in status.values() if v["status"] == "online")
    total = len(status)

    # Also print to console
    print(f"\n[check_tools] {online}/{total} online")
    for tool, info in status.items():
        icon = "[OK]" if info["status"] == "online" else "[--]"
        print(f"  {icon} {tool:30s} {info['endpoint']}  {info['status']}")

    return {
        "summary": f"{online}/{total} online",
        "online_count": online,
        "total_count": total,
        "tools": status,
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "role": "router", "tool_ports": TOOL_PORTS}


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Co-evolve Tool Router")
    print(f"Listening: http://0.0.0.0:5556")
    print(f"Tools: {json.dumps(TOOL_PORTS, indent=2)}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=5556, log_level="info")
