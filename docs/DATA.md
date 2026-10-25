# Rewritten data

The separate archive contains 41,495 unique QA records and 41,481 unique generated
images from 16 source runs. It includes all valid saved editing turns, not only
final selections or samples in a preferred difficulty band. It contains no
original seed images.

| Subset | QA records | Images |
| --- | ---: | ---: |
| Main co-evolution | 36,088 | 36,088 |
| Exploratory edits | 2,883 | 2,883 |
| Offline ablation | 2,276 | 2,264 |
| Earlier image tools | 248 | 246 |
| Total | 41,495 | 41,481 |

The metadata is `samples.jsonl`, with `id`, `image`, `question`, `answer`, and
`provenance`. Images are relative to the archive root. The archive is unchanged
by the source-code reorganization and is not stored in Git.

To use every packaged record with the original Parquet writer, adapt only the
metadata field names and call the source function:

```python
import json, random
from pathlib import Path
from build_solver_grpo_data_chartqa import samples_to_parquet

root = Path('/path/to/rewritten-dataset').resolve()
samples = []
for line in (root / 'samples.jsonl').read_text().splitlines():
    row = json.loads(line)
    image = (root / row['image']).resolve()
    assert root in image.parents and image.is_file()
    samples.append({
        'image_path': str(image), 'question': row['question'],
        'answer': str(row['answer']), 'sample_id': row['id'],
        'p_hat': row['provenance'][0].get('p_hat', -1), 'source': 'genenv',
    })
random.seed(42)
random.shuffle(samples)
samples_to_parquet(samples, 'outputs/all_rewrites/solver_grpo.parquet')
```

This performs no historical pass-rate filtering and uses the original writer's
validation behavior. `--n_original 0` in the trajectory CLI disables random
original-example mixing, but its `already_good` branch can still select seed
images. Using the packaged metadata above avoids those seed images.

Rewriter training requires trajectories and the images shown to the Planner,
including seed images for first turns. A rewritten-image archive alone cannot
reconstruct all Rewriter training inputs; supply the original training images
separately when running `build_rewriter_rwr_chartqa_data.py`.
