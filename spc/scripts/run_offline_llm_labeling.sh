#!/bin/bash
# Usage: bash scripts/run_offline_llm_labeling.sh <processed_dir> <out_cache_dir> [--save_logits true]
PROCESSED_DIR=${1:/home/carla_user/processed}
OUT_DIR=${2:/home/carla_user/results}
SAVE_LOGITS=${3:-"--save_logits false"}

python - <<'PY'
import os, json, argparse
from src.dataset.dataset import DeepAccidentDataset
from src.llm.cot_controller import CoTController

proc_dir = os.environ.get("PROCESSED_DIR", "$PROCESSED_DIR")
out_dir = os.environ.get("OUT_DIR", "$OUT_DIR")
os.makedirs(out_dir, exist_ok=True)

ds = DeepAccidentDataset(mode='train')
cot = CoTController(llm_model="gpt-4o", cache_dir=out_dir)

for i in range(len(ds)):
    sample = ds[i]
    # sample_id used to cache per-sample
    dec, parsed = cot.decide(sample, sample_id=i)
    if i % 100 == 0:
        print("Processed", i)
print("LLM offline labeling finished. Cache at", out_dir)
PY
