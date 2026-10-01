#!/usr/bin/env bash
# First quarter of the August 2026 Lichess dump -> skew-move analysis.
#
# Downloaded in parallel pieces because database.lichess.org limits each
# connection (one ran at 0.5-1.2 MB/s, four together at 3.9 MB/s); a single
# stream would have spent 2-4 hours waiting on the network. The pieces are a
# byte prefix, which decompresses up to the cut.
set -u
cd /Users/anvit/Documents/Projects/psychological-chess-engine
PY=.venv/bin/python
URL=https://database.lichess.org/standard/lichess_db_standard_rated_2026-08.pgn.zst
OUT=build/lichess_2026-08_q1.pgn.zst
BYTES=7536465590       # a quarter of 30,145,862,359
GAMES=22978081         # a quarter of 91,912,325
PARTS=8
REPORT=build/db_report.txt

fail() { echo "FAILED: $*" | tee -a "$REPORT"; touch build/db_pipeline.done; exit 1; }
: > "$REPORT"

if [ ! -s "$OUT" ]; then
  step=$(( (BYTES + PARTS - 1) / PARTS ))
  pids=()
  for i in $(seq 0 $((PARTS - 1))); do
    s=$(( i * step )); e=$(( s + step - 1 )); [ "$e" -ge "$BYTES" ] && e=$(( BYTES - 1 ))
    curl -sS --fail --retry 8 --retry-all-errors -r "$s-$e" -o "$OUT.part$i" "$URL" &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || fail "download piece $i"; done
  for i in $(seq 0 $((PARTS - 1))); do
    s=$(( i * step )); e=$(( s + step - 1 )); [ "$e" -ge "$BYTES" ] && e=$(( BYTES - 1 ))
    got=$(stat -f %z "$OUT.part$i")
    [ "$got" -eq $(( e - s + 1 )) ] || fail "piece $i is $got bytes, expected $(( e - s + 1 ))"
  done
  cat $(for i in $(seq 0 $((PARTS - 1))); do echo "$OUT.part$i"; done) > "$OUT" || fail "joining pieces"
  rm -f "$OUT".part*
fi
echo "downloaded $(stat -f %z "$OUT") bytes" | tee -a "$REPORT"

caffeinate -i $PY -m src.eval.skew_database scan --source "$OUT" --max-games "$GAMES" \
  > build/db_scan.log 2>&1 || fail "scan (see build/db_scan.log)"
tail -1 build/db_scan.log | tee -a "$REPORT"

# Four workers, not the machine's five: the live bot shares the CPU and
# budgets its moves by the clock, not by how loaded the machine is.
caffeinate -i $PY -m src.eval.skew_database evaluate --workers 4 \
  > build/db_eval.log 2>&1 || fail "evaluate (see build/db_eval.log)"

$PY -m src.eval.skew_database report >> "$REPORT" 2>&1 || fail "report"
touch build/db_pipeline.done
