# Experiment plan for Parts 2–4

Written 16 November 2026, before any agent was trained. The prediction sheet
left some details of the setup open. They are fixed here so that no choice can
be adjusted after seeing results. Deviations are dated in `WORKLOG.md`.

## 1. Setup from the prediction sheet

Breakout, mode 0, difficulty 0, frameskip 4. An episode is one life from a
fresh game, after 0–30 random paddle moves.

| Outcome | Definition | Reward |
| --- | --- | --- |
| Success | 2 bricks hit | +1, terminal |
| Failure | Ball lost | 0, terminal |
| Timeout | 2,000 agent steps | Reported separately |

| Setting | Definition |
| --- | --- |
| A | Sticky-action probability 0 |
| B | Sticky-action probability 0.25 |
| C | B with a reward of −0.001 on every step, including the last |
| D | Agents trained in B, evaluated on a layout with a 32 px gap in the wall |

Each run has 500,000 agent steps. Both agents use γ = 0.99. Q-learning starts
from a zero table with α = 0.1 and ε falling linearly from 1.0 to 0.05 over the
first half. PPO uses learning rate 3·10⁻⁴, entropy coefficient 0.01, 4 epochs,
clip 0.2 and GAE λ = 0.95.

## 2. State

Both agents get the same information. The Q-table indexes it directly; PPO
gets each component scaled to [0, 1], with an extra flag for ball in play.

| Component | Values |
| --- | --- |
| Ball y | 8 bins, boundaries at 93, 110, 130, 150, 165, 177 and 189 px (paddle top at 189) |
| Ball direction | dx and dy, each −1, 0 or +1 |
| Bricks hit | 0 or 1 |
| Paddle zone | Paddle centre left of 56 px, 56–104 px, or right of 104 px |
| Horizontal error | 11 bins of ball x − paddle centre, as in the sheet |
| Before launch | Paddle position in 8 equal bins of 17 px |

**Deviation from the sheet.** The sheet counted about 2,100 cells for nonzero
ball directions and left zero directions open. The direction is zero on at
least one axis in about 3% of steps (the first step in play and at bounces).
We keep zero as its own value instead of remembering the previous direction,
so the state has no hidden memory. This gives 8 × 9 × 2 × 3 × 11 + 8 = 4,760 cells.

## 3. Training and evaluation

| Item | Choice |
| --- | --- |
| PPO | Stable-Baselines3, MLP with 2 × 64 tanh units, 8 environments × 128 steps per rollout, 4 minibatches. The budget counts steps in all 8 environments |
| Start moves | Each is LEFT or RIGHT with equal probability. They are not agent decisions and do not count towards the budget. Their number is logged |
| Timeout | Truncation: both agents bootstrap from the next state |
| Training seeds | 0–9 in Part 2, 0–4 in Part 3, the same for both agents. Each run is seeded once |
| Evaluation seeds | 10,000–10,199. Every episode is reset with its own seed, so all agents and checkpoints meet the same 200 starts, and any episode can be replayed exactly |
| Evaluation points | Untrained, then at 5%, 10%, 20%, …, 100% of the budget. 200 episodes each, no updates |
| Evaluation policy | Greedy Q with random tie-breaks; most probable PPO action. Sampled PPO actions are reported separately at the final point |
| Transfer | The final B agents of all 10 seeds, on the gap layout (D) and on the 12 px paddle (P7), with the same evaluation seeds |
| Training curves | Success rate in consecutive windows of 10,000 steps. No other smoothing |
| Evaluation curves | Unsmoothed. Mean and between-seed SD, with every seed drawn |

In Setting A the start moves are the only source of variation, so the 200
evaluation episodes contain repeated trajectories. We report the number of
distinct trajectories next to the A results.

## 4. Tuning

Both agents get the same allowance: 6 configurations × 3 seeds × 500,000 steps
in Setting B. Tuning uses its own training seeds (100–102) and evaluation seeds
(20,000–20,199). The seeds in section 3 are never used for tuning.

| Agent | Grid |
| --- | --- |
| Q-learning | α ∈ {0.05, 0.1, 0.3} × final ε ∈ {0.05, 0.01} |
| PPO | Learning rate ∈ {1·10⁻⁴, 3·10⁻⁴, 1·10⁻³} × entropy coefficient ∈ {0, 0.01} |

Both grids contain the values from the sheet. For each agent we select the
configuration with the highest mean success rate at the final evaluation and
use it in A, B and C. On a tie, the sheet's values are kept.

## 5. If the agents do not learn

Success gives the only reward in A and B, and random play reaches 2 bricks in
1–4.5% of lives. An agent may therefore see too few successes to learn from.

- If the selected configuration of **neither** agent reaches a 10% success
  rate in tuning, we change success to **1 brick** in all settings. Random play
  reaches 1 brick in 10.5–14% of lives. The reward stays terminal, as the task
  requires for Setting A. We repeat the tuning and annotate the prediction sheet.
- If only one agent stays below 10%, nothing changes. That difference is a result.

## 6. Part 3

Setting B, 5 seeds, 500,000 steps. One factor changes at a time and all others
stay at the configuration selected in tuning. The selected value of each
factor reuses seeds 0–4 from Part 2.

| Investigation | Values |
| --- | --- |
| Exploration (Q-learning) | ε linear 1 → 0.05 over 50%; constant 0.1; linear 1 → 0.01 over 10% |
| Planning horizon (both) | γ = 0.9, 0.99, 0.999 |
| Policy exploration (PPO) | Entropy coefficient 0, 0.01, 0.05 |
| Update size | Q-learning α = 0.02, 0.1, 0.5. PPO learning rate 3·10⁻⁵, 3·10⁻⁴, 3·10⁻³ |
| PPO data reuse | 1, 4, 10 epochs per rollout |

If tuning selects a value outside a list, that value is added to the list.

The diagnostic experiment is chosen after these results. Its question and the
two explanations it separates are written in `WORKLOG.md` before it is run.

## 7. Part 4 episode selection

Episodes come from the final evaluation in Setting B, both agents, all seeds.

- **Typical successes and failures (3 + 3).** For each outcome we plot the
  episode lengths, find the most common length range and take the episodes
  closest to its median, from different agents and seeds where possible.
- **Surprising successes (2).** The successes with the lowest value estimate at
  launch (highest Q-value or PPO value), relative to the same agent's other episodes.
- **Surprising failures (2).** The failures with the highest value estimate in
  the last five steps before the ball is lost.
- **Alternative actions.** At each annotated decision we restore the saved
  emulator state and play every other action 20 times with different sticky
  noise. The outcome shares show whether the chosen action was poor or the
  outcome unlucky.
