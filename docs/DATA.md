# Rewritten visual-QA data

The separately prepared archive has 41,495 unique QA records and 41,481 unique
images. This repository contains processing code; it does not contain the archive,
original seed data, model weights, or experiment outputs.

| Subset | QA records | Images |
| --- | ---: | ---: |
| Main co-evolution | 36,088 | 36,088 |
| Exploratory edits | 2,883 | 2,883 |
| Offline ablation | 2,276 | 2,264 |
| Earlier image tools | 248 | 246 |
| Total | 41,495 | 41,481 |

`samples.jsonl` uses `id`, `image`, `question`, `answer`, and optional `provenance`.
Image paths are relative to the extracted directory. Additional files include
`images/`, subset ID lists, `summary.json`, `image_manifest.json`, and counts.
The total includes saved successful intermediate edits, not just final selections.
Some historical runs did not save edited CSVs; their images can be used for
Solver training/evaluation but are not sufficient to restart table editing.

```bash
python tools/verify_dataset.py /path/to/rewritten-dataset
python tools/prepare_rewritten_training.py \
  --dataset-root /path/to/rewritten-dataset --subset coevolution \
  --output-dir outputs/training
```

The converter defaults to `--subset all`; an explicit subset reads
`subsets/NAME.txt`. It groups source image IDs and shared generated images before
splitting. A connected group cannot occur in both train and validation. Preserve
`provenance[].original_image_id` when converting other edited datasets so edits of
the same seed remain grouped. Without provenance, grouping uses generated image
paths only. `--format jsonl` writes portable splits; Parquet resolves image paths
on the conversion machine. `--data-source` and `--system-prompt-file` configure
training metadata and instructions.

RWR requires the images actually presented to each Planner turn, including seed
images for first turns. These seed images must be obtained separately.
