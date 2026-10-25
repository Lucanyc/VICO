#!/usr/bin/env python3
"""
Evaluate the tool-augmented Solver (GRPO with tool) on VICO.

Aligns with training format produced by build_solver_tool_data.py
and train_solver_tool.sh:
  - Same TOOL_AGENT_SYSTEM_PROMPT
  - Stop generation at </tool_call>; route call through tool router at $TOOL_ROUTER_URL
  - Inject observation as user-role <tool_response>...</tool_response>
  - Loop up to --max_turns until <answer> appears
  - Send image to router via extra_fields={"images": [<abs path>]}

Usage:
    TOOL_ROUTER_URL=http://localhost:5556 \
    CUDA_VISIBLE_DEVICES=0,1 python3 evaluate_with_tool.py \
        --model_path /path/to/VICO/GRPO-with-tool/further_train_ckpt/global_step_130_merged \
        --data_json /path/to/dataset/vico/converted_test/vico_vlmgym_format.json \
        --image_root /path/to/dataset/vico \
        --tp 2 --max_turns 5 \
        --output evaluate_with_tool_step130.json
"""

import os, json, re, argparse, time, uuid
from typing import Optional
import requests
from vllm import LLM, SamplingParams
from tqdm import tqdm


# ── EXACT system prompt from build_solver_tool_data.py ──
TOOL_AGENT_SYSTEM_PROMPT = """You are a tool-augmented VLM. For every query you MUST:

1. Think about which tool to use inside <think></think>.
2. Call exactly one tool.
3. Select the tool based on what the image contains and what the question asks:
   - Need to extract structured data from a chart, table, or plot → chartmoe
   - Need to detect, locate, or count objects in a scene → sam3_detect
   - Need to read text, numbers, or labels from a document or figure → easyocr_tool
   - Need to compute a math expression → calculator
   - Need to formalize a geometry diagram → diagramformalizer

Call format:
   <tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>
   <tool_call>{"tool": "sam3_detect", "task": "detect_objects"}</tool_call>
   <tool_call>{"tool": "easyocr_tool", "task": "read_text"}</tool_call>
   <tool_call>{"tool": "calculator", "expression": "python_expr"}</tool_call>
   <tool_call>{"tool": "diagramformalizer", "task": "formalize"}</tool_call>

4. After the tool result returns in <observation></observation>, use it as a structured starting point. Combine the tool output with your own visual understanding of the image to reason step by step inside <think></think>. The tool output may be partial or imperfect — use your visual understanding to fill in gaps and verify details. Then give the final answer inside <answer></answer>."""


TOOL_CALL_RE = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
ANSWER_RE = re.compile(r'<answer>\s*(.*?)\s*</answer>', re.DOTALL)


def extract_answer(text: str) -> Optional[str]:
    if not text:
        return None
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


def has_tool_call(text: str) -> bool:
    return TOOL_CALL_RE.search(text or "") is not None


def relaxed_match(pred: Optional[str], gt: str) -> bool:
    """VICO relaxed accuracy: case-insensitive exact or numeric within 5%."""
    if pred is None:
        return False
    p, g = pred.strip().lower(), gt.strip().lower()
    if p == g:
        return True
    try:
        pf = float(p.replace(',', '').replace('%', '').replace('$', '').strip())
        gf = float(g.replace(',', '').replace('%', '').replace('$', '').strip())
        if abs(gf) < 1e-9:
            return abs(pf) < 1e-6
        return abs(pf - gf) / abs(gf) <= 0.05 + 1e-9
    except (ValueError, TypeError):
        return False


def call_tool_router(router_url: str, traj_id: str, action_text: str,
                     image_path: str, timeout: float = 120.0) -> str:
    """POST to /get_observation, return formatted observation string."""
    url = router_url.rstrip('/') + '/get_observation'
    payload = {
        "trajectory_ids": [traj_id],
        "actions": [action_text],
        "finish": [False],
        "extra_fields": [{"images": [image_path]}],
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        d = r.json()
        return str(d["observations"][0])
    except Exception as e:
        return f"Tool router error: {e}"


def build_initial_messages(image_url: str, question: str):
    return [
        {"role": "system", "content": TOOL_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": question},
        ]},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", default="evaluate_with_tool_results.json")
    parser.add_argument("--router_url", default=os.environ.get("TOOL_ROUTER_URL", "http://localhost:5556"))
    parser.add_argument("--n", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gpu_mem", type=float, default=0.85)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Strip /get_observation if user pasted full url
    args.router_url = args.router_url.replace("/get_observation", "").rstrip('/')

    # ── Health check ──
    try:
        h = requests.get(args.router_url + "/health", timeout=5).json()
        print(f"Router health: {h}")
    except Exception as e:
        print(f"[WARN] router health check failed at {args.router_url}: {e}")

    # ── Load data ──
    with open(args.data_json) as f:
        data = json.load(f)
    if args.n > 0:
        data = data[:args.n]
    print(f"Loaded {len(data)} samples")

    valid = []
    for s in data:
        img_path = os.path.join(args.image_root, s["image_path"])
        if os.path.exists(img_path):
            s["_full_image_path"] = os.path.realpath(img_path)
            valid.append(s)
    print(f"Valid samples: {len(valid)}")

    # ── Resume ──
    results = []
    correct = 0
    total = 0
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            prev = json.load(f)
        results = prev.get("results", [])
        done_ids = {r["id"] for r in results}
        correct = sum(1 for r in results if r.get("correct"))
        total = len(results)
        valid = [s for s in valid if s["id"] not in done_ids]
        print(f"Resume: {total} done, {len(valid)} remaining")

    # ── Load model ──
    print(f"Loading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=os.path.realpath(args.image_root),
    )
    # Stop at </tool_call> so we can intercept and inject observation,
    # OR at <|im_end|> via natural EOS so <answer>...</answer> can complete.
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=1.0,
        stop=["</tool_call>"],
    )

    t0 = time.time()

    for i in tqdm(range(0, len(valid), args.batch_size), desc="Evaluating"):
        batch = valid[i:i + args.batch_size]
        # Per-sample state: messages (full chat history), assembled response text,
        # done flag, turn counter, traj_id (for tool router).
        states = []
        for s in batch:
            img_url = f"file://{s['_full_image_path']}"
            states.append({
                "sample": s,
                "messages": build_initial_messages(img_url, s["question"]),
                "assistant_text": "",   # full assistant text accumulated across turns
                "done": False,
                "turn": 0,
                "traj_id": f"vico_eval_{uuid.uuid4().hex[:12]}",
                "tool_calls": [],
                "tool_outputs": [],
            })

        for turn in range(args.max_turns):
            active_idx = [k for k, st in enumerate(states) if not st["done"]]
            if not active_idx:
                break

            # ── Generate this turn for all active samples ──
            msgs_batch = [states[k]["messages"] for k in active_idx]
            outputs = llm.chat(msgs_batch, sampling_params=sampling_params)

            for k, out in zip(active_idx, outputs):
                gen = out.outputs[0]
                resp = gen.text
                stop_reason = gen.stop_reason  # the matched stop string, or None
                st = states[k]
                st["turn"] = turn + 1

                # Re-attach </tool_call> when stop matched on it
                stopped_at_tool = False
                if stop_reason == "</tool_call>" or (resp.rfind("<tool_call>") > resp.rfind("</tool_call>")):
                    if not resp.rstrip().endswith("</tool_call>"):
                        resp = resp + "</tool_call>"
                    stopped_at_tool = True

                st["assistant_text"] += resp
                # Persist this assistant turn into the chat history
                st["messages"].append({"role": "assistant", "content": resp})

                # Final answer? Stop.
                if extract_answer(resp) is not None:
                    st["done"] = True
                    continue

                # Tool call? Route, append observation, continue.
                if stopped_at_tool and has_tool_call(resp):
                    obs = call_tool_router(
                        args.router_url, st["traj_id"], resp,
                        st["sample"]["_full_image_path"],
                    )
                    st["tool_calls"].append(TOOL_CALL_RE.findall(resp)[-1].strip())
                    st["tool_outputs"].append(obs[:1000])
                    # Match training-time qwen3_vl wrapping: user role with <tool_response>
                    st["messages"].append({
                        "role": "user",
                        "content": f"<tool_response>\n{obs}\n</tool_response>",
                    })
                    continue

                # No tool call and no answer → stop
                st["done"] = True

        # ── Score and save batch ──
        for st in states:
            s = st["sample"]
            full_resp = st["assistant_text"]
            pred = extract_answer(full_resp)
            gt = str(s["answer"])
            ok = relaxed_match(pred, gt)
            correct += int(ok)
            total += 1
            results.append({
                "id": s["id"],
                "question": s["question"],
                "image_path": s["image_path"],
                "gt": gt,
                "pred": pred,
                "correct": ok,
                "turns": st["turn"],
                "tool_calls": st["tool_calls"],
                "tool_outputs": st["tool_outputs"],
                "response": full_resp,
            })

        acc = correct / total * 100 if total else 0.0
        avg_tools = sum(len(r["tool_calls"]) for r in results) / max(total, 1)
        tqdm.write(f"  [{total}] acc={acc:.1f}%  avg_tools={avg_tools:.2f}")

        # Incremental save
        with open(args.output, "w") as f:
            json.dump({
                "model": args.model_path,
                "router_url": args.router_url,
                "max_turns": args.max_turns,
                "total": total,
                "correct": correct,
                "accuracy": round(acc, 2),
                "results": results,
            }, f, ensure_ascii=False, indent=2)

    # ── Summary ──
    elapsed = time.time() - t0
    acc = correct / total * 100 if total else 0
    n_tool_called = sum(1 for r in results if r["tool_calls"])
    n_with_answer = sum(1 for r in results if r["pred"] is not None)
    avg_tools = sum(len(r["tool_calls"]) for r in results) / max(total, 1)

    print(f"\n{'='*60}")
    print(f"VICO (with-tool) Evaluation")
    print(f"  Model:        {args.model_path}")
    print(f"  Router:       {args.router_url}")
    print(f"  Samples:      {total}")
    print(f"  Accuracy:     {correct}/{total} = {acc:.2f}%")
    print(f"  Has tool:     {n_tool_called}/{total} ({n_tool_called/max(total,1)*100:.1f}%)")
    print(f"  Has answer:   {n_with_answer}/{total} ({n_with_answer/max(total,1)*100:.1f}%)")
    print(f"  Avg tools:    {avg_tools:.2f}")
    print(f"  Time:         {elapsed:.0f}s ({elapsed/max(total,1):.1f}s/sample)")
    print(f"{'='*60}")

    with open(args.output, "w") as f:
        json.dump({
            "model": args.model_path,
            "router_url": args.router_url,
            "max_turns": args.max_turns,
            "total": total,
            "correct": correct,
            "accuracy": round(acc, 2),
            "has_tool_call": n_tool_called,
            "has_answer": n_with_answer,
            "avg_tools_per_sample": round(avg_tools, 3),
            "time_s": round(elapsed, 1),
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
