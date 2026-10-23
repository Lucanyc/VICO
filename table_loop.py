#!/usr/bin/env python3
"""
Planner-Executor-Solver Loop — VICO GenEnv version

Flow per sample:
  1. Load CSV table + question + answer + answer_cell
  2. Rewriter decides modification direction
  3. GPT-5.2 modifies CSV data (outputs new table as JSON)
  4. Extract new_answer from modified CSV at answer_cell position
  5. Render new chart via matplotlib
  6. Solver evaluates (new_chart, question, new_answer) → p̂

Answer correctness:
  - Tier 1: new_answer = value at answer_cell in modified CSV (100% clean)
  - Tier 2: answer unchanged, only rendering style modified
"""
import os, sys, json, re, argparse, random, time
from PIL import Image
from typing import Dict, List
from collections import Counter

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from table_editor import TableEditor


def atomic_json_save(data, filepath):
    tmpf = filepath + ".tmp"
    with open(tmpf, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        try: os.fsync(f.fileno())
        except OSError: pass
    try: os.replace(tmpf, filepath)
    except OSError:
        with open(filepath, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def verify_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False


# === Answer checking ===
def clean_answer(text):
    if not text: return ""
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if m: return m.group(1).strip()
    for pat in [r'\\boxed\{', r'\\boxed \{']:
        m2 = re.search(pat, text)
        if m2:
            s, d, i = m2.end(), 1, m2.end()
            while i < len(text) and d > 0:
                if text[i] == '{': d += 1
                elif text[i] == '}': d -= 1
                i += 1
            if d == 0: return text[s:i-1].strip()
    text = text.strip()
    for p in ["The answer is:", "The answer is", "Answer:", "Answer"]:
        if text.lower().startswith(p.lower()): text = text[len(p):].strip(); break
    text = re.sub(r'[\\%$]', '', text.rstrip('.')).strip()
    # Remove all unit/modifier words (not just at end)
    text = re.sub(r'\b(million|billion|trillion|thousand|percent|approximately|about|around|nearly|over|'
                  r'kg|km|miles|meters|dollars|euros|pounds|yen|yuan|rupees|'
                  r'australian|american|british|canadian|chinese|indian|japanese|'
                  r'us|usd|eur|gbp|cad|aud)\b',
                  '', text, flags=re.IGNORECASE).strip()
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def strip_thinking(text):
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    return re.sub(r'<think>[\s\S]*$', '', text).strip()

def check_correct(gt, pred):
    gt_s = str(gt).strip().lower()
    pred_s = clean_answer(str(pred)).lower()
    if not pred_s: return False
    if gt_s == pred_s: return True
    # Numeric comparison with scale awareness (million/billion)
    try:
        gn = float(gt_s.replace(',','').replace('%',''))
        pn = float(pred_s.replace(',','').replace('%',''))
        # Direct match (0.5% tolerance)
        if gn != 0 and abs(gn - pn) / abs(gn) <= 0.005:
            return True
        # Check if pred is gt * scale factor (e.g., 4.2 million = 4200000)
        for scale in [1e3, 1e6, 1e9]:
            if gn != 0 and abs(gn * scale - pn) / abs(gn * scale) <= 0.005:
                return True
            if pn != 0 and abs(pn * scale - gn) / abs(pn * scale) <= 0.005:
                return True
    except: pass
    return False


# === Solver ===
class SolverEvaluator:
    SYSTEM_PROMPT = (
        "Please answer the question about the chart image. "
        "Think step by step in <think> tags, then give your final answer in <answer> tags."
    )

    def __init__(self, model_path, n_rollouts=8, temperature=0.7):
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor
        self.n_rollouts = n_rollouts
        orig_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        device_list = orig_devices.split(",") if orig_devices else []
        solver_gpu = device_list[1] if len(device_list) >= 2 else "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = solver_gpu
        print(f"Loading Solver: {model_path} (on GPU {solver_gpu})")
        self.llm = LLM(model=model_path, tensor_parallel_size=1, max_model_len=4096,
                        trust_remote_code=True, gpu_memory_utilization=0.80)
        os.environ["CUDA_VISIBLE_DEVICES"] = orig_devices
        self.sampling_params = SamplingParams(temperature=temperature, max_tokens=1024, n=n_rollouts)
        self.processor = AutoProcessor.from_pretrained(model_path)
        print(f"Solver ready (n_rollouts={n_rollouts})")

    def evaluate(self, image_path, question, answer):
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": self.SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question}]}]
        prompt = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        try:
            pil = Image.open(image_path).convert("RGB")
            max_pixels = 4096 * 4096
            w, h = pil.size
            if w * h > max_pixels:
                ratio = (max_pixels / (w * h)) ** 0.5
                pil = pil.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        except Exception as e:
            print(f"    Cannot open image: {e}")
            return 0.0

        try:
            outs = self.llm.generate([{"prompt": prompt, "multi_modal_data": {"image": pil}}],
                                     self.sampling_params)
        except ValueError as e:
            print(f"    Image incompatible: {str(e)[:80]}")
            return 0.0

        correct = 0
        for i, c in enumerate(outs[0].outputs):
            raw = strip_thinking(c.text)
            pred = clean_answer(raw)
            ok = check_correct(answer, raw)
            if ok: correct += 1
            print(f"    rollout {i}: pred='{pred}' gt='{answer}' correct={ok}")
        p = correct / self.n_rollouts
        print(f"    → {correct}/{self.n_rollouts} correct, p̂={p:.3f}")
        return p


# === Main Loop ===
def run_loop(rewriter_planner, generator, solver, sample,
             tables_dir, png_dir, output_dir,
             max_turns=5, target_lo=0.35, target_hi=0.65):

    sample_id = sample.get("_id", "unknown")
    imgname = sample["imgname"]
    question = sample["query"]
    original_answer = str(sample["label"])
    tier = sample.get("tier", 2)
    answer_cell = sample.get("answer_cell")

    basename = os.path.splitext(imgname)[0]
    csv_path = os.path.join(tables_dir, f"{basename}.csv")
    img_path = os.path.join(png_dir, imgname)

    traj = {
        "sample_id": sample_id, "imgname": imgname,
        "question": question[:100], "original_answer": original_answer,
        "tier": tier, "answer_cell": answer_cell,
        "turns": [], "final_phat": None, "final_image_path": None,
        "final_answer": None, "status": "pending", "total_turns": 0,
    }

    if not os.path.exists(img_path):
        traj["status"] = "failed"
        traj["error"] = f"Image not found: {img_path}"
        return traj

    if not os.path.exists(csv_path):
        traj["status"] = "failed"
        traj["error"] = f"CSV not found: {csv_path}"
        return traj

    # Evaluate original
    print(f"  [Original] Evaluating...")
    p0 = solver.evaluate(img_path, question, original_answer)
    traj["p_hat_original"] = p0

    if target_lo <= p0 <= target_hi:
        traj.update({"status": "already_good", "final_phat": p0,
                     "final_image_path": img_path, "final_answer": original_answer})
        return traj

    # Chain-based modification
    history = []
    cur_p = p0
    current_answer = original_answer

    for turn in range(1, max_turns + 1):
        print(f"  [Turn {turn}] Planner (p̂={cur_p:.3f})...")

        # Get direction from Rewriter
        plan = rewriter_planner.get_plan(
            image_path=img_path, question=question, answer=current_answer,
            p_hat_original=cur_p, scene_objects=[], history=history)

        edit_instruction = plan.get("params", {}).get("edit_instruction", "Modify chart data")
        reasoning = plan.get("reasoning", "")
        print(f"  [Turn {turn}] Direction: {edit_instruction[:60]}...")

        # Generate: modify CSV + render + extract answer
        mod_path = os.path.join(output_dir, f"{sample_id}_turn{turn}.png")
        gen_result = generator.generate(
            csv_path=csv_path, question=question,
            original_answer=current_answer,
            answer_cell=answer_cell, tier=tier,
            edit_instruction=edit_instruction,
            output_path=mod_path, p_hat=cur_p,
        )

        if not gen_result["success"]:
            print(f"  [Turn {turn}] Failed: {gen_result['error']}")
            traj["turns"].append({
                "turn": turn, "plan": plan, "exec_success": False,
                "exec_error": gen_result["error"],
                "new_answer": None, "p_hat": None,
            })
            history.append({"tool": "chart_rerender", "params": plan.get("params", {}),
                            "p_hat": None, "error": gen_result["error"]})
            continue

        if not verify_image(mod_path):
            print(f"  [Turn {turn}] Image verification failed")
            continue

        new_answer = gen_result["new_answer"]
        n_before = gen_result["n_rows_before"]
        n_after = gen_result["n_rows_after"]
        print(f"  [Turn {turn}] Rows: {n_before}→{n_after}, answer: {current_answer}→{new_answer}")

        # Evaluate
        print(f"  [Turn {turn}] Evaluating...")
        ph = solver.evaluate(mod_path, question, str(new_answer))
        print(f"  [Turn {turn}] → p̂={ph:.3f}", end="")

        traj["turns"].append({
            "turn": turn, "plan": plan, "exec_success": True,
            "modified_image": mod_path,
            "new_answer": new_answer,
            "n_rows_before": n_before, "n_rows_after": n_after,
            "p_hat": ph,
        })
        history.append({"tool": "chart_rerender",
                        "params": plan.get("params", {}),
                        "p_hat": ph, "new_answer": new_answer})

        if target_lo <= ph <= target_hi:
            print(f" ✓ TARGET HIT!")
            traj.update({"status": "success", "final_phat": ph,
                         "final_image_path": mod_path,
                         "final_answer": new_answer, "total_turns": turn})
            return traj
        else:
            print(f" ({'too_easy' if ph > target_hi else 'too_hard'})")

        cur_p = ph
        current_answer = new_answer

    # Pick best
    valid = [t for t in traj["turns"] if isinstance(t.get("p_hat"), float)]
    if valid:
        best = min(valid, key=lambda t: abs(t["p_hat"] - 0.5))
        traj.update({"status": "partial", "final_phat": best["p_hat"],
                     "final_image_path": best.get("modified_image"),
                     "final_answer": best.get("new_answer"),
                     "total_turns": max_turns})
    else:
        traj.update({"status": "failed", "total_turns": max_turns})
    return traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json", required=True,
                        help="VICO rewriter JSON (with tier and answer_cell)")
    parser.add_argument("--tables_dir", required=True,
                        help="Directory with CSV tables")
    parser.add_argument("--png_dir", required=True,
                        help="Directory with original PNG charts")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--model_path", required=True,
                        help="Solver model path")
    parser.add_argument("--n_rollouts", type=int, default=8)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--target_lo", type=float, default=0.35)
    parser.add_argument("--target_hi", type=float, default=0.65)
    # Rewriter (planner)
    parser.add_argument("--planner_mode", default="local", choices=["local", "gpt5", "simple"])
    parser.add_argument("--planner_model", default="gpt-5", help="GPT model for gpt5 planner mode")
    parser.add_argument("--rewriter_model_path", default=None)
    parser.add_argument("--rewriter_gpu_memory", type=float, default=0.80)
    # Output
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.data_json) as f:
        all_data = json.load(f)
    samples = all_data[args.start:args.start + args.n]
    # Add IDs
    for i, s in enumerate(samples):
        s["_id"] = f"chart_{args.start + i}"
    print(f"Samples: {len(samples)} (index {args.start}-{args.start+args.n-1})")

    trajs = []
    done = set()
    if args.resume and os.path.exists(args.output_json):
        with open(args.output_json) as f:
            trajs = json.load(f)
        done = set(t["sample_id"] for t in trajs)
        print(f"Resume: {len(trajs)} done")

    todo = [(s["_id"], s) for s in samples if s["_id"] not in done]
    print(f"To process: {len(todo)}")

    if todo:
        # Initialize planner
        original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        gpu_list = [g.strip() for g in original_cuda.split(",") if g.strip()] if original_cuda else []

        if args.planner_mode == "local" and args.rewriter_model_path:
            if len(gpu_list) >= 2:
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list[0]
            from local_planner import LocalPlanner
            planner = LocalPlanner(
                model_path=args.rewriter_model_path,
                gpu_memory_utilization=args.rewriter_gpu_memory,
                max_model_len=4096,
            )
        elif args.planner_mode == "gpt5":
            from api_planner import APIPlanner
            planner = APIPlanner(model=args.planner_model)
        else:
            # Fallback: simple planner that returns generic instructions
            class SimplePlanner:
                def get_plan(self, **kwargs):
                    p = kwargs.get("p_hat_original", 0.5)
                    history = kwargs.get("history", [])

                    # Dynamic instruction based on p̂ and history
                    if p > 0.85:
                        instr = (f"p̂={p:.3f} is very high. Make the chart MUCH harder: "
                                 f"add many rows with near-miss values, change the answer value, "
                                 f"rearrange data to create visual confusion")
                    elif p > 0.65:
                        instr = (f"p̂={p:.3f} is too high. Add a few distractor rows "
                                 f"with similar values to increase difficulty")
                    elif p < 0.15:
                        instr = (f"p̂={p:.3f} is very low. Simplify significantly: "
                                 f"remove confusing rows, make the answer value more distinct")
                    elif p < 0.35:
                        instr = (f"p̂={p:.3f} is too low. Simplify slightly: "
                                 f"reduce the number of similar values")
                    else:
                        instr = f"p̂={p:.3f} is near target. Make minimal adjustments"

                    # Add history context
                    if history:
                        prev_phats = [h.get("p_hat") for h in history if isinstance(h.get("p_hat"), float)]
                        if prev_phats:
                            too_hard = sum(1 for ph in prev_phats if ph < 0.35)
                            too_easy = sum(1 for ph in prev_phats if ph > 0.65)
                            if too_hard > 0:
                                instr += f". Previous {too_hard} attempts were too aggressive, use FEWER changes"
                            if too_easy > 0:
                                instr += f". Previous {too_easy} attempts were too mild, use MORE changes"

                    return {"tool": "chart_rerender",
                            "params": {"edit_instruction": instr},
                            "reasoning": f"p_hat={p:.3f}, history={len(history)} turns"}
            planner = SimplePlanner()
            print("Using SimplePlanner (no local rewriter)")

        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda

        generator = TableEditor()
        solver = SolverEvaluator(args.model_path, args.n_rollouts)

        for idx, (sid, s) in enumerate(todo):
            print(f"\n{'─'*60}\n[{idx+1}/{len(todo)}] {sid}")
            print(f"  Q: {s['query'][:70]}...\n  A: {s['label']}, tier={s.get('tier', '?')}")
            tr = run_loop(planner, generator, solver, s,
                          args.tables_dir, args.png_dir, args.output_dir,
                          args.max_turns, args.target_lo, args.target_hi)
            trajs.append(tr)

            fp = tr.get('final_phat')
            fa = tr.get('final_answer', '?')
            fp_str = f"{fp:.3f}" if isinstance(fp, float) else "?"
            print(f"  → {tr['status']} | p̂: {tr.get('p_hat_original','?')} → {fp_str} | answer: {s['label']} → {fa}")

            atomic_json_save(trajs, args.output_json)
            print(f"  💾 {len(trajs)} saved")

    # Summary
    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    if not trajs:
        print("  No trajectories.")
        return

    sc = Counter(t["status"] for t in trajs)
    fps = [t["final_phat"] for t in trajs if isinstance(t.get("final_phat"), float)]
    sweet = sum(1 for p in fps if .35 <= p <= .65) if fps else 0
    print(f"  Total: {len(trajs)}, Status: {dict(sc)}")
    if fps:
        print(f"  Sweet zone: {sweet}/{len(fps)} ({sweet/len(fps)*100:.1f}%)")
    print(f"  Saved: {args.output_json}")


if __name__ == "__main__":
    main()
