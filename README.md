# psychological-chess-engine

LatentBlunder, a Lichess bot ([@LatentBlunder](https://lichess.org/@/LatentBlunder))
built to make *humans* blunder. Each move is an expectimax over Maia-3's
prediction of the human reply. Stockfish sets a safety floor, so the bot only
plays a trap when the trap costs little against best defence. Against other
bots it skips the human model and plays Stockfish's best move.

## Setup

```bash
python3.13 -m venv .venv
.venv/bin/pip install --no-deps maia2==0.10.0
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pip install git+https://github.com/CSSLab/maia3   # AGPL-3.0, not on PyPI
.venv/bin/maia3-cache --model maia3-5m
echo "LICHESS_API_TOKEN=lip_..." > .env   # bot:play scope; gitignored
```

You need Stockfish on `PATH` (or in `bin/stockfish`) and the `.pth` weights in
`models/`. The weights are not in git; regenerate them with
`src.training.distill` and `src.training.train_dpo`.

## Run

| Command | What it does |
|---|---|
| `python -m src.lichess.bot` | Lichess bridge. Accepts human challenges, rated 1100–1700, blitz and rapid. `--allow-bots --challenge-bots` also plays bots. |
| `python -m src.main --rating 1500` | Local Pygame board against the engine |
| `python -m src.eval.arena` | Offline matches between arms against Maia opponents |
| `python -m src.eval.game_review` | Reviews every game the account has played, per move, at depth 16 |
| `python -m src.eval.clinical_trials` | Measures the randomized arms (pacing, narrow paths, traps) on the human games |
| `python -m src.eval.live_analyzer` | Telemetry from the live game log, grouped by arm |

Headless deployment: `Dockerfile` and `deploy/setup_systemd.sh`.

## Layout

- `src/engine`: search, evaluators (Stockfish, Maia-1/2/3), opening books
- `src/lichess`: bridge, outbound challenger, move pacing
- `src/training`: DPO and distillation, skew mining, repertoire builder
- `src/eval`: arena, database and clock studies, game review, dashboards
- `src/ui`: Pygame board, mining and telemetry views
- `docs`: research notes and game reviews
- `build`: run output (gitignored, except the mined skew positions)

## Checks

```bash
.venv/bin/python -m pytest -q
mypy --strict src
```
