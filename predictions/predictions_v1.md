# Assignment 4, Part 1: Prediction sheet

**Eivind Systad Geiran and Trym Andreas Johnsen** · IT3212 · Option 2: Breakout

| Field | Value |
|---|---|
| Version | v1, original. Frozen at git tag `predictions-v1` and not edited after submission |
| Written | 2026-11-12, before any agent was trained. Frozen: *(time filled in at freeze)* |
| Authors | First draft by Claude (AI), edited and approved by Eivind, reviewed by Trym (see `AI_RECORD.md`) |
| Done before writing | Environment inspection only (`src/inspect_env.py`): random actions and a fixed ball-tracking rule, no learning. Results in `results/inspection/` |

Later interpretations go in `predictions/annotations.md`, dated and keyed to the IDs below.

## 0. Provisional setup

This is our current interpretation. Part 2 may change details. Any change is recorded as an annotation.

| Item | Provisional choice |
|---|---|
| Agents | Tabular one-step Q-learning (ε-greedy) and clipped PPO (small MLP actor-critic) |
| State | Both see the same binned RAM features: ball in play; ball x, ball y and paddle x (8 bins each); ball direction (dx, dy). About 2,000 cells. PPO gets the bin values scaled to [0, 1], so the comparison is about the learning methods, not their inputs |
| Episode | One life from a fresh game, after 0–30 random paddle moves drawn from the episode seed |
| Outcomes | **Success:** 2 bricks hit (terminal, reward +1). **Failure:** ball lost (reward 0). **Timeout:** 2,000 agent steps, reported separately |
| Settings | **A:** sticky actions off (deterministic). **B:** sticky probability 0.25. **C:** B plus a step penalty c = 0.001. **D:** B agents evaluated with the 12 px paddle (difficulty 1) instead of 16 px |
| Budget | Same for both agents, counted in agent steps (1 step = 4 frames). Provisionally 500k steps per run. **Small budget** = the first 10% |
| Defaults | γ = 0.99 for both. Q-learning: α = 0.1, ε falls linearly from 1.0 to 0.05 over the first half. PPO: learning rate 3·10⁻⁴, entropy 0.01, 4 epochs, clip 0.2, GAE λ = 0.95. Both get the same tuning allowance |
| Evaluation | Separate evaluation seeds. 200 episodes every 10% of the budget, with no updates. Primary policy: greedy Q (ties broken at random) and most-probable PPO action. Sampled PPO is reported separately |
| Seeds | At least 5 training seeds per agent, the same list for both |
| **Decision rule** | Each metric is computed per training seed. **Supported:** the mean difference is in the predicted direction and larger than the pooled between-seed SD. **Contradicted:** larger than the pooled SD in the opposite direction. **Inconclusive:** otherwise |

Inspection facts used below: random play reaches 2 bricks in 1% (A) and 4.5% (B) of lives, with a median of 27 steps. The tracking rule reaches 2 bricks in 100% (A) and 89% (B) of lives. In A, different seeds give identical trajectories for the same actions.

## 1. Predictions

### P1. Which agent improves fastest with a small interaction budget?
- **Prediction:** PPO. At 10% of the budget, PPO's evaluation success rate is higher than Q-learning's in both A and B.
- **Why:** Success needs one skill: keeping the paddle under the ball. That is a smooth function of ball x minus paddle x. PPO's network can transfer what it learns at one paddle position to neighbouring positions, and GAE spreads the delayed +1 over many earlier steps in each update. The Q-table learns each of about 2,000 cells separately and moves value back only one step per visit. Q-learning can reuse its exploratory data off-policy, but we expect generalisation to matter more here.
- **Test:** In A and B, evaluation success rate at 10% of the budget, plus the area under the success curve over that first 10%.
- **Contradicted if:** Q-learning's success rate at 10% is higher by more than the pooled seed SD in A or B.

### P2. Which agent shows the largest performance variation across seeds?
- **Prediction:** Q-learning. At the full budget, its between-seed SD of evaluation success rate is at least 1.5 times PPO's, in both A and B.
- **Why:** With a sparse reward, a Q-learning seed only starts to learn after ε-greedy play happens to succeed. Greedy evaluation of a partly filled table is also all-or-nothing: one wrong argmax in a common cell repeats in every episode. PPO changes its policy in small clipped steps and shares what it learns across similar states, so its seeds should drift less apart. The deterministic dynamics in A should make the all-or-nothing effect stronger.
- **Test:** SD and range across at least 5 seeds of the final evaluation success rate, in A and B. Supported if SD_Q ≥ 1.5 × SD_PPO in both settings. Contradicted if SD_PPO ≥ 1.5 × SD_Q in either setting. Inconclusive otherwise. An SD from 5 seeds is itself noisy, so we also report the ranges.
- **Contradicted if:** PPO has the larger spread.

### P3. How will stochastic transitions change the routes agents learn?
In Breakout, a route is how the paddle moves relative to the ball.

- **Prediction:** Compared with agents trained in A, agents trained in B (a) succeed less often, (b) are already under the ball earlier when it comes down, and (c) switch actions less often. This holds for both algorithms.
- **Why:** With probability 0.25 the previous action is repeated, so a last-moment correction can fail and rapid left-right switching is unreliable. A paddle that is already under the falling ball is robust to that. In A exact timing always works, so nothing pushes an agent to be early. Even the tracking rule drops from 100% to 89% success, so B is harder for a sensible policy too.
- **Test:** In evaluation, per seed: success rate; mean |ball x − paddle centre| while the ball is falling in the lower part of the field (ball y ≥ 150); and the share of steps whose action differs from the previous step's. A vs B, per agent.
- **Contradicted if:** B agents are not closer to the ball in the lower field, or they switch at least as often as A agents.

### P4. How does a step penalty change success rate and episode length?
- **Prediction:** With c = 0.001 per step (C vs B): (a) failed episodes are shorter in C, for both agents; (b) PPO's success rate is lower in C; (c) Q-learning's success rate changes little (inconclusive under our rule); (d) successful episodes are slightly shorter in C.
- **Why:** Until an agent succeeds often, the only reward it sees in C is the penalty, and the quickest way to stop the penalty is to lose the ball. C therefore rewards giving up early. That is not optimal behaviour: failing fast only beats succeeding when c × (success length − failure length) > 1, roughly c > 0.007 here, and 0.001 is far below that. A drop in success would therefore be a learning effect, not the optimum. PPO follows this gradient with its own data and can settle there. A zero-initialised Q-table becomes optimistic once the actions it has tried turn negative, which pushes it towards untried actions and may cancel the effect.
- **Test:** Success and failure rates (not returns, which use different rewards), and median lengths of failed and of successful episodes. B vs C, per agent.
- **Contradicted if:** failed episodes are not shorter in C, or PPO's success rate in C is at least as high as in B.

### P5. Will the agent with the highest training return also perform best in evaluation?
- **Prediction:** Not reliably. In at least one of A, B and C, the order by training return and the order by evaluation success differ. Q-learning's evaluation success exceeds its training success by more than PPO's does.
- **Why:** Q-learning trains with ε-greedy actions, and the random actions lose balls, so its training return understates its greedy policy. The cliff-walking example in Sutton & Barto shows the same effect: the online return of Q-learning is lower than its greedy policy deserves. PPO's training return comes from its own sampled policy, so evaluating its most probable action changes less. It can even hurt, if the sampling was doing useful work.
- **Test:** Per seed: success rate over the last 10% of training episodes against the final evaluation success rate. In each of A, B and C, rank the two agents by mean final training return and by mean evaluation success.
- **Contradicted if:** the two rankings agree in all three settings, and Q-learning's evaluation-minus-training gap is not larger than PPO's.

### P6. Which agent would we initially recommend, and what would change our mind?
- **Prediction:** For repeated operation under uncertainty (B and C) we initially recommend PPO, because we expect it to learn faster (P1), vary less between seeds (P2) and cope better with a changed paddle (P7). For a quick classroom demonstration we expect tabular Q-learning to be better, because its table can be read directly and each run is cheap.
- **We would change our mind if:**
    1. Q-learning's failure rate in B is lower than PPO's by more than the pooled seed SD at the full budget.
    2. PPO's worst seed is worse than Q-learning's worst seed in B. Reliability matters more than the mean.
    3. At equal wall-clock time, Q-learning is ahead because PPO's cost per step is much higher.
    4. P7 is contradicted.

### P7 (extra). What survives the change to a narrower paddle (Setting D)?
- **Prediction:** Both B agents succeed less often with the 12 px paddle, but PPO loses fewer percentage points than Q-learning.
- **Why:** The RAM layout is unchanged, so the features keep their meaning and the Q-table still applies. But a ball at the edge of a 16 px paddle misses a 12 px one. If S2 holds, Q-learning has learned to accept edge catches and PPO has learned a margin. The tracking rule drops from 89% to 72%.
- **Test:** Success in B minus success in D for the same trained agents, per seed.
- **Contradicted if:** Q-learning's drop is smaller than PPO's by more than the pooled seed SD.

## 2. States where we expect the approaches to differ

<p>
<img src="../results/inspection/frames/S1_ball_not_launched.png" width="150" alt="S1">
<img src="../results/inspection/frames/S2_ball_falling_at_paddle_edge.png" width="150" alt="S2">
<img src="../results/inspection/frames/S3_ball_rising_mid_field.png" width="150" alt="S3">
</p>

*Examples of S1, S2 and S3 (left to right), captured during inspection with the tracking rule.*

For each state we will query the Q-table and the PPO policy at every feature cell that matches the definition. In evaluation episodes we will also log the chosen action, Q-values and policy probabilities whenever the state occurs. Part 4 uses the same logs.

### S1. Ball not launched
*Definition:* ball not in play (RAM bytes 99 and 101 are 0), any paddle position. Only FIRE makes progress.

- **Q-learning:** FIRE in the paddle positions that are common at launch (central bins). In rarely visited edge bins the values are still tied near zero, so the launch is delayed.
- **PPO:** FIRE is the most probable action in every paddle position, with P(FIRE) > 0.9.
- **Why:** Every episode starts here, but the random start moves mostly leave the paddle near the centre. The table learns each cell separately, while the network can generalise "ball not in play → FIRE" across paddle positions.
- **Contradicted if:** Q-learning's greedy action is FIRE in all 8 paddle bins for every seed, or PPO's P(FIRE) < 0.5 in any bin. We also compare the number of steps from the start to the launch.

### S2. Ball falling at the paddle edge
*Definition:* ball falling (dy > 0), ball y ≥ 165, and ball x over the paddle's outer 4 px (4–8 px from the paddle centre).

- **Q-learning:** NOOP. It stays, because the edge will catch the ball.
- **PPO:** moves towards the ball, centring the paddle under it.
- **Why:** Q-learning is off-policy. It learns the value of acting greedily afterwards, and under that assumption an edge catch is as good as a centred one. PPO learns the value of its own stochastic policy. With its sampled actions and sticky repeats, an edge position sometimes ends in a miss, so PPO prefers a margin. This is the off-policy/on-policy contrast in cliff walking (Sutton & Barto, Example 6.6).
- **Contradicted if:** at ball contact, Q-learning's mean |ball x − paddle centre| is not larger than PPO's, or PPO's most probable action in S2 cells is NOOP for most seeds.

### S3. Ball rising in mid-field
*Definition:* ball rising (dy < 0), 100 ≤ ball y ≤ 150, 50 ≤ ball x ≤ 110. The next paddle contact is many steps away, so the action matters little for now.

- **Q-learning:** no stable preference. The greedy action changes between checkpoints and seeds, and often between consecutive steps.
- **PPO:** a stable choice, typically moving the paddle towards the ball's x position. Consecutive actions rarely switch.
- **Why:** The four Q-values are almost equal here, so small estimation errors decide the argmax, and a table has nothing to smooth them with. PPO's network shares parameters with nearby states where the action does matter, so its preference here follows those.
- **Contradicted if:** Q-learning's action-switch rate in S3 is not higher than PPO's, or Q-learning's seeds agree on the greedy action in S3 cells as often as PPO's seeds (agreement: at least 4 of 5 seeds choose the same action).

## 3. Dissent (Trym)

*Trym adds here any prediction he disagrees with, as D1, D2, …, using the same fields (Prediction, Why, Test, Contradicted if). The predictions above are not changed.*

## 4. Change policy

This sheet is not edited after submission. When our interpretation changes, we add a dated entry to `predictions/annotations.md`. The entry names the prediction ID and says what changed and why. Results are compared with this original text.

## References

- Machado, M. C. et al. (2018). Revisiting the Arcade Learning Environment. *JAIR* 61. (Sticky actions, p = 0.25.)
- Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
- Sutton, R. S. and Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. (Q-learning; Example 6.6, cliff walking.)
