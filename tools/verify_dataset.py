#!/usr/bin/env python3
"""Validate counts, image checksums, sample IDs, and portable references."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path)
    args = ap.parse_args()
    root = args.dataset.resolve()
    summary = json.loads((root / "summary.json").read_text())
    manifest = json.loads((root / "image_manifest.json").read_text())
    files = set()
    pixels = set()
    for row in manifest:
        relative = row["image"]
        path = (root / relative).resolve()
        assert root in path.parents and relative.startswith("images/"), relative
        assert relative not in files and row["pixel_sha256"] not in pixels
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], relative
        files.add(relative)
        pixels.add(row["pixel_sha256"])
    ids, referenced = set(), set()
    with (root / "samples.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            assert row["id"] not in ids and row["question"] and str(row["answer"]).strip()
            assert row["image"] in files and row["provenance"]
            ids.add(row["id"])
            referenced.add(row["image"])
    assert referenced == files
    assert len(ids) == summary["unique_qa_samples"]
    assert len(files) == summary["unique_generated_images"]
    assert summary["exported_original_images"] == 0
    assert {"images/" + p.name for p in (root / "images").iterdir()} == files
    for subset in (root / "subsets").glob("*.txt"):
        selected = subset.read_text().splitlines()
        assert len(selected) == len(set(selected)) and set(selected) <= ids
    print(json.dumps({"verified_qa_samples": len(ids), "verified_unique_images": len(files),
                      "image_checksums": "passed", "relative_paths": "passed"}))


if __name__ == "__main__":
    main()
