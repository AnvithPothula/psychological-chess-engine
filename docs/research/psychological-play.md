# Deep Research: Beating humans through psychology rather than strength

## Executive Summary

The evidence says the strongest lever against humans is not engine strength but
**which positions humans are made to face**. Across millions of online and
grandmaster moves, the difficulty of the position predicts a blunder far better
than the player's rating or remaining clock, and in our own August 2026 Lichess
sample, quiet positions with two or fewer safe replies were blundered 7 times as
often as easy ones. The clock matters too, but only near the end: blunder rates
are flat until roughly ten seconds remain, then jump, and low time compounds with
an ambiguous position.

The bots that beat strong humans most convincingly do it from objectively lost
positions. Leela's odds bots give up a knight or a queen and still perform at
strong-GM and 2000-2700 levels against people, by being trained against a model of
human play (Maia) and told to assume the opponent is weaker. That is the template
for this project: objective soundness can be traded for practical chances, within
limits, and the trade is worth more against humans than against engines.

One hard limit: Lichess flags bots that lose on purpose. A bot may be made weaker,
but it must always be trying to win.

## Key Findings

1. **Position difficulty dominates human error.** A difficulty measure (blunder
   potential, the share of legal moves that are blunders) predicts whether a human
   blunders with 0.73-0.75 accuracy, against 0.55 for rating and 0.53 for time
   left; each +0.2 of blunder potential adds more to the blunder rate than 600
   rating points. Blunder rates rise smoothly from 0.0024 to 0.42 across blunder
   potential in grandmaster games. Some positions are "skill-anomalous": stronger
   players blunder them *more*.
   ([Anderson, Kleinberg, Mullainathan, TKDD](https://sendhil.org/wp-content/uploads/2019/08/Publication-13.pdf);
   [KDD 2016 slides](https://www.cs.toronto.edu/~ashton/slides/chesserrors-kdd2016-slides.pdf))
2. **Time only bites near zero, and it compounds with difficulty.** Blunder rates
   are flat (5.5-5.8%) above about 10 seconds remaining and near 11.6% at zero.
   In elite blitz the rate reaches about 24% under 10 seconds against under 5% at
   120-180 seconds, and low time combined with an ambiguous position produces a
   sharp jump. Moves that take longer are blunders more often (instant 4%, over
   3 seconds 8%+), because hard positions take time.
   ([Anderson et al.](https://sendhil.org/wp-content/uploads/2019/08/Publication-13.pdf);
   [Scientific Reports, 2026](https://www.nature.com/articles/s41598-026-59689-z))
3. **Under time pressure, people play safer, especially when behind.** Less
   thinking time leads professional players to more risk-averse moves, measured
   by the spread of outcomes a move allows.
   ([J. Econ. Behav. & Org., 2025](https://www.sciencedirect.com/science/article/pii/S0167268125003373))
4. **Practical play beats material deficits.** LeelaQueenOdds was trained on data
   generated with Maia 1900; queen-odds Leela performs at 2000-2700 depending on
   time control and knight-odds Leela at or above strong-GM level. Its live
   settings include Contempt 450, WDLCalibrationElo 3300 and Temperature 0.8.
   Contempt in Lc0 means "assume I am this many Elo stronger", which makes it play
   more aggressively and avoid draws; Komodo's Armageddon mode gains about 30 Elo
   from the same idea.
   ([LeelaQueenOdds](https://github.com/notune/LeelaQueenOdds);
   [Piece-odds challenge](https://lczero.org/blog/2024/12/the-leela-piece-odds-challenge-what-does-it-take-you-to-win-against-leela/);
   [Odds settings](https://lczero.org/blog/2024/02/update-on-playing-with-piece-odds-against-lc0-on-lichess/);
   [WDL contempt](https://lczero.org/blog/2023/07/the-lc0-v0.30.0-wdl-rescale/contempt-implementation/);
   [Sadler on contempt](https://matthewsadler.me.uk/engine-chess/setting-up-wdl-contempt-for-leela-in-nibbler/);
   [Komodo](https://chessprogramming.org/Komodo))
5. **Strength can be calibrated in a human way.** Allie spends search time where
   humans would think longer and lands within 49 Elo of opponents from 1000 to
   2600; Maia-2 predicts moves from both players' skill.
   ([Allie](https://arxiv.org/abs/2410.03893); [Maia-2](https://arxiv.org/html/2409.20553v1))
6. **Tilt exists but is modest.** The previous game's result predicts the next
   (b = 0.25 in log-odds over about a million Lichess games), fading over about
   seven games; streaks are more frequent than chance, and beginners are
   streakier than strong players.
   ([Devine](http://seandevine.org/blog/chessBlog.html);
   [NDpatzer, citing Chowdhary et al. 2023](https://lichess.org/@/NDpatzer/blog/science-of-chess-winning-streaks-losing-streaks-and-skill/K4NmnE6b))
7. **Bots must not lose on purpose.** A Lichess bot was flagged for "losing some
   games in a few moves on purpose"; random-move bots are marked too, so nobody can
   farm rating from them. Bot rules forbid sandbagging and boosting.
   ([Lichess forum](https://lichess.org/forum/lichess-feedback/violation-of-terms-of-service-for-my-bot);
   [Lichess API spec](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/lichess-api.yaml))

## Detailed Analysis

How each lever maps to this bot, and what changed:

| lever | evidence | before | now, against humans |
|---|---|---|---|
| position difficulty | findings 1-2, our clock analysis | narrow-path term off | on, quiet positions only (omega 10) |
| clock | finding 2 | ignored | narrow paths count double when the human is under 30s |
| risk for practical chances | finding 4 | floor -180cp | -250cp, sliding to -450 with a large expected payout |
| won positions | our bot games | full "keep it won" rule | stay above +3, but traps still allowed |
| opening | our games | trap book everywhere | trap book against humans only |
| tilt | finding 6 | accepts rematches | unchanged; no chat, no taunting |

Measured in this repo's arena against maia3@1900, 300 games per configuration:

| configuration | score | opponent blunder rate | plies to decisive |
|---|---|---|---|
| previous bot | 0.998 | 11.98% | 22.2 |
| human profile, full depth | 0.997 | 11.74% | 24.3 |
| human profile, depth capped at 6 (live) | 0.990 | 13.48% | 24.5 |

The depth cap, not the profile, moved Maia's blunder rate (+3.2 sigma against the
uncapped profile), so the live bot plays humans at depth 6. Against Maia the score
barely moves, so the arena cannot say how much weaker the cap makes it against
people; live games will.

Against bots the bot keeps the conservative settings: engines are not fooled by
any of this, and the trap bets cost five won games against 2000+ engines.

## Contrarian Views And Risks

- **Correlation is not steering.** Humans blunder in difficult positions, but in
  this project steering into high blunder potential did not raise Maia's blunder
  rate in the arena. Humans may respond differently; live volume is too low to
  tell yet.
- **The odds bots have strength to spare.** Leela underneath is far beyond any
  human, so its practical risks are backed by deep calculation. A weaker engine
  taking the same risks has less margin.
- **Shallow search is an unhuman way to weaken.** Cutting depth makes the bot miss
  tactics a human would see, which can look bot-like. Allie's human-time search is
  the principled alternative, at a much larger build cost.
- **Low-clock humans simplify** (finding 3), so narrow positions may be hardest
  to create exactly when the clock boost wants them.
- **Tilt is small for strong players**, and anything beyond accepting rematches
  (chat, provocation) would break Lichess rules and is not on the table.

## Open Questions

- Does the bot's own move speed change how long humans think, or how often they
  blunder? No study was found.
- Does narrow-path steering raise human blunder rates live? Needs a randomized
  comparison with real opponents.
- How does arena strength against Maia translate to strength against people?

## Sources

- https://sendhil.org/wp-content/uploads/2019/08/Publication-13.pdf: Anderson, Kleinberg, Mullainathan, human error vs difficulty, skill and time
- https://www.cs.toronto.edu/~ashton/slides/chesserrors-kdd2016-slides.pdf: KDD 2016 slides, "difficulty is the dominant feature"
- https://www.nature.com/articles/s41598-026-59689-z: Scientific Reports 2026, blunders vs remaining time and ambiguity in elite blitz
- https://www.sciencedirect.com/science/article/pii/S0167268125003373: time pressure makes professionals risk-averse
- https://arxiv.org/abs/2410.03893: Allie, human-aligned chess with time-adaptive search
- https://arxiv.org/html/2409.20553v1: Maia-2, skill-aware human move prediction
- https://github.com/notune/LeelaQueenOdds: LeelaQueenOdds network, trained with Maia 1900 data
- https://lczero.org/blog/2024/12/the-leela-piece-odds-challenge-what-does-it-take-you-to-win-against-leela/: odds bots' performance levels
- https://lczero.org/blog/2024/02/update-on-playing-with-piece-odds-against-lc0-on-lichess/: LeelaKnightOdds live settings
- https://lczero.org/blog/2023/07/the-lc0-v0.30.0-wdl-rescale/contempt-implementation/: Lc0 WDL contempt
- https://matthewsadler.me.uk/engine-chess/setting-up-wdl-contempt-for-leela-in-nibbler/: contempt explained by a GM
- https://chessprogramming.org/Komodo: Komodo Armageddon mode, about +30 Elo
- https://www.lesswrong.com/posts/eQvNBwaxyqQ5GAdyx/some-data-from-leelapieceodds: LeelaPieceOdds data (graphs only; training against Maia mentioned)
- http://seandevine.org/blog/chessBlog.html: tilt and hype in about a million Lichess games
- https://lichess.org/@/NDpatzer/blog/science-of-chess-winning-streaks-losing-streaks-and-skill/K4NmnE6b: streaks and skill, citing Chowdhary et al. 2023
- https://lichess.org/forum/lichess-feedback/violation-of-terms-of-service-for-my-bot: bot flagged for losing on purpose
- https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/lichess-api.yaml: bot rules (no pools, no sandbagging)
- https://github.com/lichess-bot-devs/lichess-bot/wiki/Configure-lichess-bot: lichess-bot configuration and daily limits

## Rerun Inputs

workflow: firecrawl-deep-research
topic: designing a chess bot to beat humans through psychological and practical means while capping objective strength
depth: thorough
output: markdown
