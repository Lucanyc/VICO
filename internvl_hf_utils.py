"""
Shared HF utilities for InternVL inference.
Bypasses vLLM (which produces gibberish on certain merged checkpoints).
"""

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size=448):
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.Resize((input_size, input_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, w, h, image_size):
    best_diff = float("inf")
    best = (1, 1)
    area = w * h
    for ratio in target_ratios:
        ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - ar)
        if diff < best_diff or (
            diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]
        ):
            best_diff = diff
            best = ratio
    return best


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    w, h = image.size
    ar = w / h
    ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    tar = find_closest_aspect_ratio(ar, ratios, w, h, image_size)
    tw, th = image_size * tar[0], image_size * tar[1]
    blocks = tar[0] * tar[1]
    resized = image.resize((tw, th))
    out = []
    for i in range(blocks):
        box = (
            (i % (tw // image_size)) * image_size,
            (i // (tw // image_size)) * image_size,
            ((i % (tw // image_size)) + 1) * image_size,
            ((i // (tw // image_size)) + 1) * image_size,
        )
        out.append(resized.crop(box))
    if use_thumbnail and len(out) > 1:
        out.append(image.resize((image_size, image_size)))
    return out


def load_image_tiles(path, input_size=448, max_num=6, dtype=torch.bfloat16):
    """Single image -> [N_tiles, 3, H, W] tensor on cuda."""
    image = Image.open(path).convert("RGB")
    tiles = dynamic_preprocess(image, image_size=input_size, max_num=max_num)
    tfm = build_transform(input_size)
    return torch.stack([tfm(t) for t in tiles]).to(dtype).cuda()


def load_image_tiles_multi(paths, input_size=448, max_num=6, dtype=torch.bfloat16):
    """Multi-image -> (concatenated_tensor, num_patches_per_image)."""
    tfm = build_transform(input_size)
    pixels = []
    num_patches_list = []
    for p in paths:
        image = Image.open(p).convert("RGB")
        tiles = dynamic_preprocess(image, image_size=input_size, max_num=max_num)
        pixels.append(torch.stack([tfm(t) for t in tiles]))
        num_patches_list.append(len(tiles))
    return torch.cat(pixels, dim=0).to(dtype).cuda(), num_patches_list


def load_model(model_path, system_prompt=None):
    """Load InternVL model + tokenizer with flash-attn if available."""
    print(f"Loading tokenizer from {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    try:
        import flash_attn  # noqa: F401
        attn_kw = {"attn_implementation": "flash_attention_2"}
        print("flash-attn detected, enabling flash_attention_2")
    except ImportError:
        attn_kw = {}
        print("flash-attn NOT available; using default (eager) attention — slow")
    print("Loading model weights...")
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True,
        **attn_kw,
    ).cuda().eval()
    if system_prompt:
        model.system_message = system_prompt
    print("Model ready.")
    return tok, model


def chat_single(model, tok, image_path, query, gen_cfg, max_num=6):
    """One image + text -> response string. Adds <image>\\n prefix automatically."""
    pixel = load_image_tiles(image_path, max_num=max_num)
    full_q = "<image>\n" + query if "<image>" not in query else query
    return model.chat(tok, pixel, full_q, gen_cfg)


def chat_multi(model, tok, image_paths, query, gen_cfg, max_num=6):
    """Multi-image + text -> response. Auto-prepends Image-i: <image>\\n markers."""
    pixel, num_patches_list = load_image_tiles_multi(image_paths, max_num=max_num)
    if "<image>" not in query:
        prefix = "".join(f"Image-{i+1}: <image>\n" for i in range(len(image_paths)))
        full_q = prefix + query
    else:
        full_q = query
    return model.chat(tok, pixel, full_q, gen_cfg,
                      num_patches_list=num_patches_list)


def make_gen_cfg(tok, max_new_tokens=2048, temperature=0.0):
    cfg = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        pad_token_id=tok.eos_token_id,
    )
    if temperature > 0:
        cfg["temperature"] = temperature
        cfg["top_p"] = 0.95
    return cfg
