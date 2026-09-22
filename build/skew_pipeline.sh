#!/usr/bin/env bash
# Wait for the mine, compile the book, check the confound, run the arena.
set -u
cd /Users/anvit/Documents/Projects/psychological-chess-engine
PY=.venv/bin/python
REPORT=build/skew_report.txt
: > "$REPORT"

say() { echo "$@" | tee -a "$REPORT"; }

# 1. wait for the miner to exit (it has its own 3h timeout)
while pgrep -f "src.training.skew_miner" > /dev/null; do sleep 30; done
say "=== MINE ==="
say "$(tail -1 build/skew_mine.log)"
say "rate limits: $(grep -c '429' build/skew_mine.log)"

if [ ! -s build/skew_positions.jsonl ]; then
  say "MINE PRODUCED NOTHING -- stopping here."
  touch build/skew_pipeline.done; exit 1
fi
say "entries: $(wc -l < build/skew_positions.jsonl)"

# 2. confound check on the 1100-1700 band
say ""
say "=== CONFOUND CHECK (1100-1700) ==="
$PY - <<'EOF' 2>&1 | tee -a build/skew_report.txt
import json, statistics as st
rows=[json.loads(l) for l in open("build/skew_positions.jsonl")]
def corr(xs, ys):
    n=len(xs); mx,my=st.fmean(xs),st.fmean(ys); sx,sy=st.pstdev(xs),st.pstdev(ys)
    return 0.0 if not sx or not sy else sum((a-mx)*(b-my) for a,b in zip(xs,ys))/n/(sx*sy)
sk=[r["skew"] for r in rows]
print(f"n = {len(rows)}   mean skew {st.fmean(sk):+.4f}   range {min(sk):+.4f}..{max(sk):+.4f}")
for name, key in (("average_rating","average_rating"), ("evaluation_cp","evaluation_cp")):
    r = corr(sk, [x[key] for x in rows])
    print(f"  corr(skew, {name:<14}) = {r:+.3f}   r^2 = {r*r:.3f}   explains {100*r*r:.0f}%")
for gate in (25, 10):
    keep=[x for x in rows if abs(x["evaluation_cp"])<gate]
    if keep: print(f"  |eval|<{gate:>3}cp: {len(keep):>4} kept, mean skew {st.fmean(x['skew'] for x in keep):.4f}")
print("  top 5:")
for x in sorted(rows,key=lambda z:-z["skew"])[:5]:
    print(f"    ply {x['ply']:>2} {x['san']:<6} {x['bot_color']:<5} skew {x['skew']:+.3f} "
          f"eval {x['evaluation_cp']:+4d} avg {x['average_rating']} n={x['games']:>9,}")
EOF

# 3. compile the book
say ""
say "=== COMPILE ==="
$PY -m src.training.polyglot_compiler --input build/skew_positions.jsonl \
    --output src/engine/books/skew.bin 2>&1 | tee -a "$REPORT"

# 4. powered arena
say ""
say "=== ARENA: standard vs skew, 1000 games/arm ==="
$PY -m src.eval.arena --arms standard skew --games 1000 --rating 1500 --plies 60 \
    --output build/arena_skew.jsonl 2>&1 | grep -vE "^loading|^Maia3 ready|^resolving|Warning:" | tee -a "$REPORT"

touch build/skew_pipeline.done
