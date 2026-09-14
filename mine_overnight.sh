#!/usr/bin/env bash
# Fill out the DPO dataset. Two rollouts at a time: each owns a Stockfish + lc0
# pair (~400MB), and a 21-hour run was observed at 1.37GB of lc0 alone.
# Separate --output per process: parallel appends to one file can interleave.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p build

run() {  # run <games> <rating> <tag>
  $PY -m src.training.dpo_generator \
      --games "$1" --rating "$2" --max-plies 30 \
      --output "build/pairs_$3.jsonl" --log-level INFO \
      > "build/mine_$3.log" 2>&1
}

echo "pass 1/2  high bands (1700, 1900) — these are empty today"
run 6000 1700 1700 &
run 6000 1900 1900 &
wait

echo "pass 2/2  balancing the low and mid bands"
run 5000 1100 1100 &
run 5000 1500 1500 &
wait

echo "merging"
$PY -m src.training.dedup --input build/dpo_pairs.jsonl build/pairs_*.jsonl \
                          --output build/dpo_dataset_clean.jsonl
wc -l build/dpo_dataset_clean.jsonl
