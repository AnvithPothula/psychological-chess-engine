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
# Paired: every game starts at a mined position with the bot to move, the same
# position in both arms, two passes with different Maia seeds. Opening from the
# initial position the skew book fired in 2/40 games, because nothing steers the
# bot into plies 7-10 of the mined lines. caffeinate because the mine ran 17h on
# a 3h timeout: the Mac slept and suspended it.
say "=== ARENA: standard vs skew, paired from mined positions ==="
N=$(( 2 * $($PY -c 'import json;print(len({json.loads(l)["fen"] for l in open("build/skew_positions.jsonl")}))') ))
caffeinate -i $PY -m src.eval.arena --arms standard skew --games "$N" --rating 1500 --plies 60 \
    --openings build/skew_positions.jsonl \
    --output build/arena_skew.jsonl 2>&1 | grep -vE "^loading|^Maia3 ready|^resolving|Warning:" | tee -a "$REPORT"
$PY - <<'EOF' 2>&1 | tee -a "$REPORT"
import json
rows=[json.loads(l) for l in open("build/arena_skew.jsonl")]
a={r["game"]:r for r in rows if r["arm"]=="standard"}; b={r["game"]:r for r in rows if r["arm"]=="skew"}
changed=[g for g in a if a[g]["first_move"]!=b[g]["first_move"]]
print(f"\n  first move changed by the skew book in {len(changed)}/{len(a)} pairs")
for name in ("opponent_blunders","max_opponent_error","plies"):
    d=[b[g][name]-a[g][name] for g in changed]
    m=sum(d)/len(d); sd=(sum((x-m)**2 for x in d)/(len(d)-1))**.5
    print(f"  changed pairs only, skew-standard {name:<20} {m:+.3f}  ({m/(sd/len(d)**.5):+.1f} sigma, paired)")
EOF

touch build/skew_pipeline.done
