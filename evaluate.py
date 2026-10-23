#!/usr/bin/env python3
"""Evaluate visual question answering with configurable schema, prompt and metric."""
import os, json, re, argparse, time
from pathlib import Path
from tqdm import tqdm
from vico.data import load_samples, resolve_image, save_json
from vico.metrics import PROMPTS, extract_answer, matches


# -------- edit this prompt to your liking ----------------------------------
SYSTEM_PROMPT = PROMPTS["boxed"]


# Model shortcuts. Extend as you add more local checkpoints.
MODELS_DIR = os.environ.get("VICO_MODELS_DIR", "./models")

MODEL_ALIASES = {
    "qwen3vl8b":       os.path.join(MODELS_DIR, "Qwen3-VL-8B-Instruct"),
    "qwen2.5vl7b":     os.path.join(MODELS_DIR, "Qwen2.5-VL-7B-Instruct"),
    "internvl3-8b":    os.path.join(MODELS_DIR, "InternVL3-8B-Instruct"),
    "openvlthinker7b": os.path.join(MODELS_DIR, "OpenVLThinker-7B"),
}


def resolve_model(arg: str) -> tuple[str, str]:
    """Return (model_path, tag). `arg` is an alias or a filesystem path."""
    if arg in MODEL_ALIASES:
        return MODEL_ALIASES[arg], arg
    if Path(arg).exists():
        for tag, p in MODEL_ALIASES.items():
            if os.path.realpath(p) == os.path.realpath(arg):
                return arg, tag
        return arg, Path(arg).name
    raise SystemExit(
        f"unknown model: {arg}\n"
        f"  known aliases: {', '.join(MODEL_ALIASES)}\n"
        f"  or pass a local path that exists."
    )


def mm_kwargs_for(model_path: str) -> dict:
    """Qwen-VL uses min/max_pixels; InternVL's processor rejects those
    kwargs — let it read defaults from the model config instead.

    Detects Qwen-VL family by either dir-name ('qwen*vl*') OR by reading
    config.json's `model_type` (catches Qwen2.5-VL fine-tunes whose dir
    name doesn't contain 'qwen', e.g. OpenVLThinker-7B).
    """
    name = Path(model_path).name.lower()
    is_qwen_vl = "qwen" in name and "vl" in name
    if not is_qwen_vl:
        cfg = Path(model_path) / "config.json"
        if cfg.exists():
            try:
                model_type = json.load(open(cfg)).get("model_type", "")
                is_qwen_vl = model_type.startswith("qwen") and "vl" in model_type
            except Exception:
                pass
    if is_qwen_vl:
        return {
            "min_pixels": 4 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        }
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--model_path", dest="model", required=True)
    parser.add_argument("--data", "--data_json", dest="data", required=True)
    parser.add_argument("--image-root", "--image_root", dest="image_root", required=True)
    parser.add_argument("--field-map", type=Path, help="JSON mapping canonical fields to source columns")
    parser.add_argument("--output", default="outputs/evaluation.json")
    parser.add_argument("--prompt-style", choices=list(PROMPTS), default="boxed")
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--metric", choices=["exact", "relaxed"], default="relaxed")
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--n", type=int, default=-1)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=8)
    parser.add_argument("--max-tokens", "--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--enforce-eager", action="store_true", help="Skip CUDA graphs for small smoke runs")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_tokens < 1 or args.tolerance < 0:
        parser.error("Invalid batch size, token limit, or tolerance")
    model, model_tag = resolve_model(args.model)
    field_map = json.loads(args.field_map.read_text()) if args.field_map else None
    data = load_samples(args.data, field_map)
    if args.n > 0:
        data = data[:args.n]
    for sample in data:
        sample["full_image"] = str(resolve_image(sample, args.image_root))
    if not data:
        parser.error("No evaluation samples selected")
    prompt = args.system_prompt_file.read_text() if args.system_prompt_file else PROMPTS[args.prompt_style]
    config = {"model": str(Path(model).resolve()), "data": str(Path(args.data).resolve()),
              "prompt": prompt, "prompt_style": args.prompt_style, "metric": args.metric,
              "tolerance": args.tolerance, "max_tokens": args.max_tokens,
              "temperature": args.temperature, "field_map": field_map,
              "image_root": str(Path(args.image_root).resolve())}
    output = Path(args.output)
    results = []
    if output.exists():
        if not args.resume:
            parser.error("Output already exists; use --resume or another output path")
        previous = json.loads(output.read_text())
        if previous.get("config") != config:
            parser.error("Resume configuration differs from the saved run")
        results = previous["results"]
    selected = {s["id"] for s in data}
    if any(r["id"] not in selected for r in results):
        parser.error("Saved results include samples outside this selection")
    done = {r["id"] for r in results}
    todo = [s for s in data if s["id"] not in done]
    if not todo:
        print(f"Complete: {len(results)} saved samples; no model load needed")
        return
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_memory_utilization, dtype="bfloat16",
              trust_remote_code=True, max_model_len=args.max_model_len,
              limit_mm_per_prompt={"image": 1},
              allowed_local_media_path=str(Path(args.image_root).resolve()),
              mm_processor_kwargs=mm_kwargs_for(model), enforce_eager=args.enforce_eager)
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
    start = time.time()
    for index in tqdm(range(0, len(todo), args.batch_size), desc="Evaluating"):
        batch = todo[index:index + args.batch_size]
        messages = [[{"role": "system", "content": prompt},
                     {"role": "user", "content": [
                         {"type": "image_url", "image_url": {"url": Path(s["full_image"]).as_uri()}},
                         {"type": "text", "text": s["question"]}]}] for s in batch]
        outputs = llm.chat(messages, sampling_params=sampling)
        if len(outputs) != len(batch):
            raise RuntimeError("Inference returned the wrong number of outputs")
        for sample, out in zip(batch, outputs):
            response = out.outputs[0].text
            prediction = extract_answer(response, args.prompt_style)
            results.append({"id": sample["id"], "question": sample["question"],
                            "gt": sample["answer"], "pred": prediction,
                            "correct": matches(prediction, sample["answer"], args.metric, args.tolerance),
                            "response": response})
        correct = sum(r["correct"] for r in results)
        save_json(output, {"config": config, "total": len(results), "correct": correct,
                           "accuracy": 100 * correct / len(results), "results": results})
    print(f"Completed {len(results)} samples in {time.time() - start:.1f}s: {output}")


if __name__ == "__main__":
    main()
