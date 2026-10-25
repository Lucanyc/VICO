#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Start all tool servers for Co-evolve GRPO training
#
# Each tool runs in its own conda env on a dedicated GPU+port.
# The router dispatches requests to the correct server.
#
# Usage:
#   bash start_tool_servers.sh          # start all
#   bash start_tool_servers.sh router   # start router only
#   bash start_tool_servers.sh stop     # stop all
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

VICO_DIR=${VICO_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
VERL_TOOL_DIR="${VERL_TOOL_DIR:-${VICO_DIR}/vendor/verl-tool}"
LOG_DIR="${VICO_DIR}/tool_server_logs"
mkdir -p "${LOG_DIR}"

# ─── Stop mode ───
if [ "${1:-}" = "stop" ]; then
    echo "Stopping all tool servers..."
    pkill -f "tool_router.py" 2>/dev/null && echo "  router stopped" || echo "  router not running"
    pkill -f "verl_tool.servers.serve" 2>/dev/null && echo "  servers stopped" || echo "  servers not running"
    exit 0
fi

echo "═══════════════════════════════════════════════════"
echo "  Co-evolve Tool Servers"
echo "═══════════════════════════════════════════════════"

# ─── 1. Calculator (no GPU, any env) ───
echo ""
echo "Starting Calculator (port 6700, no GPU)..."
cd ${VERL_TOOL_DIR}
CUDA_VISIBLE_DEVICES="" nohup python -m verl_tool.servers.serve \
    --host localhost --port 6700 --tool_type calculator \
    > ${LOG_DIR}/calculator.log 2>&1 &
echo "  PID: $!, log: ${LOG_DIR}/calculator.log"

# ─── 2. EasyOCR (needs GPU, easyocr env) ───
echo ""
echo "Starting EasyOCR (port 6758, GPU 4)..."
cd ${VERL_TOOL_DIR}
VERL_APPLY_QWEN25VL_PATCH=0 CUDA_VISIBLE_DEVICES=4 \
nohup conda run -n easyocr python -m verl_tool.servers.serve \
    --host localhost --port 6758 --tool_type easyocr_tool \
    > ${LOG_DIR}/easyocr.log 2>&1 &
echo "  PID: $!, log: ${LOG_DIR}/easyocr.log"

# ─── 3. ChartMoE (needs GPU, chartmoe env) ───
echo ""
echo "Starting ChartMoE (port 6658, GPU 5)..."
cd ${VERL_TOOL_DIR}
VERL_APPLY_QWEN25VL_PATCH=0 CUDA_VISIBLE_DEVICES=5 \
nohup conda run -n chartmoe python -m verl_tool.servers.serve \
    --host localhost --port 6658 --tool_type chartmoe \
    > ${LOG_DIR}/chartmoe.log 2>&1 &
echo "  PID: $!, log: ${LOG_DIR}/chartmoe.log"

# ─── 4. DiagramFormalizer (needs GPU, verl-tool env) ───
echo ""
echo "Starting DiagramFormalizer (port 7866, GPU 6)..."
cd ${VERL_TOOL_DIR}
VERL_APPLY_QWEN25VL_PATCH=0 CUDA_VISIBLE_DEVICES=6 \
nohup python -m verl_tool.servers.serve \
    --host localhost --port 7866 --tool_type diagramformalizer \
    > ${LOG_DIR}/diagramformalizer.log 2>&1 &
echo "  PID: $!, log: ${LOG_DIR}/diagramformalizer.log"

# ─── 5. Pixel Reasoner / Crop+Zoom (no GPU) ───
echo ""
echo "Starting Pixel Reasoner (port 6900, no GPU)..."
cd ${VERL_TOOL_DIR}
CUDA_VISIBLE_DEVICES="" nohup python -m verl_tool.servers.serve \
    --host localhost --port 6900 --tool_type pixel_reasoner \
    > ${LOG_DIR}/pixel_reasoner.log 2>&1 &
echo "  PID: $!, log: ${LOG_DIR}/pixel_reasoner.log"

# ─── 6. SAM3 (needs GPU, sam3/base env) ───
# NOTE: sam3 tool server needs to be written or use base env
# Uncomment when sam3 tool server is ready:
# echo ""
# echo "Starting SAM3 (port 6800, GPU 7)..."
# CUDA_VISIBLE_DEVICES=7 nohup conda run -n sam3 python -m verl_tool.servers.serve \
#     --host localhost --port 6800 --tool_type sam3 \
#     > ${LOG_DIR}/sam3.log 2>&1 &

# ─── 7. Router (no GPU) ───
if [ "${1:-}" = "router" ] || [ "${1:-all}" = "all" ] || [ -z "${1:-}" ]; then
    echo ""
    echo "Starting Router (port 5556)..."
    cd ${VICO_DIR}
    pip install aiohttp 2>/dev/null || true
    nohup python tool_router.py > ${LOG_DIR}/router.log 2>&1 &
    echo "  PID: $!, log: ${LOG_DIR}/router.log"
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "  All servers starting. Check logs in: ${LOG_DIR}/"
echo ""
echo "  Port mapping:"
echo "    5556  - Router (dispatches to tools)"
echo "    6658  - ChartMoE (extract_table)"
echo "    6700  - Calculator (math eval)"
echo "    6758  - EasyOCR (text detection)"
echo "    6900  - Pixel Reasoner (crop/zoom)"
echo "    7866  - DiagramFormalizer (geometry CDL)"
echo ""
echo "  GPU allocation:"
echo "    GPU 0-3: GRPO training (Solver)"
echo "    GPU 4:   EasyOCR"
echo "    GPU 5:   ChartMoE"
echo "    GPU 6:   DiagramFormalizer"
echo "    GPU 7:   SAM3 (when enabled)"
echo ""
echo "  Health check: curl http://localhost:5556/health"
echo "═══════════════════════════════════════════════════"
