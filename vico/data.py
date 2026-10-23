"""Load a common visual-QA schema from JSON or JSONL."""
import json
from pathlib import Path


def load_samples(path, field_map=None):
    """Return id, image, question, answer, and optional backend metadata.

    field_map maps canonical names to source column names. Original columns are
    retained, allowing an editor to consume task-specific metadata.
    """
    path = Path(path)
    with path.open() as f:
        records = ([json.loads(line) for line in f if line.strip()]
                   if path.suffix == ".jsonl" else json.load(f))
    if not isinstance(records, list):
        raise ValueError("Dataset must be a JSON array or JSONL records")
    aliases = {"id": ("id", "sample_id", "_id"),
               "image": ("image", "image_path", "imgname"),
               "question": ("question", "query"),
               "answer": ("answer", "label")}
    field_map = field_map or {}
    unknown = set(field_map) - set(aliases)
    if unknown:
        raise ValueError(f"Unknown canonical fields: {sorted(unknown)}")
    samples, seen = [], set()
    for index, record in enumerate(records):
        sample = dict(record)
        for field, names in aliases.items():
            if field in field_map:
                names = (field_map[field],)
            value = next((record[n] for n in names if n in record), None)
            if value is None and field == "id":
                value = f"sample_{index}"
            if value is None or isinstance(value, (list, dict)):
                raise ValueError(f"Record {index}: {field} must be a scalar")
            sample[field] = str(value)
        if not sample["image"] or not sample["question"]:
            raise ValueError(f"Record {index}: empty image or question")
        if sample["id"] in seen:
            raise ValueError(f"Duplicate sample id: {sample['id']}")
        seen.add(sample["id"])
        samples.append(sample)
    return samples


def resolve_image(sample, image_root):
    root = Path(image_root).resolve()
    image = (root / sample["image"]).resolve()
    if root not in image.parents or not image.is_file():
        raise ValueError(f"Missing image or image outside image_root: {sample['image']}")
    return image


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)
