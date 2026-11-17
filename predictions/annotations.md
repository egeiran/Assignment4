# Annotations to the prediction sheet

The submitted sheet, `predictions_v1.md` / `.pdf` (merge commit `61bcabe`,
submitted 13 November 2026), is preserved unchanged. This file holds dated
notes keyed to its prediction IDs: changes of setup, verdicts under the sheet's
decision rule, and changes of interpretation. A note never replaces the
original prediction.

Verdicts use the sheet's rule: **supported**, **contradicted**, **inconclusive**
or **untested**, classified per component for multi-part predictions. The
pooled between-seed SD of two groups is √((SD₁² + SD₂²) / 2). Every number below
is a mean over 10 trained agents per algorithm, with the pooled SD in brackets.

An inconclusive comparison is not read as "no effect". We state its direction:
**weak evidence for** the prediction, **weak evidence against** it, or **no
measurable difference** when the difference is below a tenth of the pooled SD.
In all three cases the difference is smaller than the variation between seeds.

All verdicts with their calculations are in
`results/analysis/verdicts/verdicts.md` (`src/verdicts.py`, `src/probes.py`).

## Setup changes

- **2026-11-16, selected configurations.** Tuning in Setting B (plan, section 4)
  selected Q-learning α = 0.3 with final ε = 0.01, and PPO learning rate 1·10⁻³
  with entropy 0.01. The sheet's defaults were α = 0.1 with final ε = 0.05, and
  learning rate 3·10⁻⁴ with entropy 0.01. Settings A, B and C use the selected
  values; the sheet's values are run in Part 3.
- **2026-11-16, state size.** The state has 4,760 cells, not "about 2,100": a
  zero ball direction is kept as its own value (plan, section 2).
- **2026-11-16, open details fixed.** Bin boundaries, PPO implementation, seeds,
  tuning and evaluation points are fixed in `plans/part2_3_plan.md`, committed
  before any training.
- **2026-11-14, Setting D.** The team confirmed the sheet's 32 px gap layout as
  D. The sheet contains no prediction for this layout (P7 concerns the narrower
  paddle). D results are therefore reported without a prior prediction.

## P1

- **2026-11-17, verdict.** Success rate at 10% of the budget (50,000 steps).
  - A: PPO 16.6%, Q-learning 17.4% (25.5). Inconclusive: no measurable difference.
  - B: PPO 21.6%, Q-learning 11.1% (23.8). Inconclusive: weak evidence for.
  - At this checkpoint the spread between seeds is larger than either mean.

## P2

- **2026-11-17, verdict.** Between-seed SD of the final success rate.
  - A: PPO 2.1, Q-learning 0.0. Supported under the sheet's threshold. The
    result rests on one PPO seed at 93.5%; the other 19 agents reach 100%.
  - B: PPO 17.5, Q-learning 16.7, ratio 1.05. Inconclusive: no measurable difference.

## P3

- **2026-11-17, verdict for (a).** Final success, A-trained in A against B-trained in B.
  - Q-learning: 100.0% against 75.9% (11.8). Supported.
  - PPO: 99.4% against 87.8% (12.4). Inconclusive: weak evidence for.
- **2026-11-17, verdict for (b).** Mean distance between ball and paddle centre
  in descent at y ≥ 150, A-trained against B-trained agents at the same p.
  - Q-learning: 11.5 px against 18.3 px at p = 0 (6.1), contradicted; 23.8 px
    against 21.9 px at p = 0.25 (5.4), weak evidence for. Overall contradicted:
    without sticky actions the B-trained agents keep the paddle further from the ball.
  - PPO: 9.1 px against 11.8 px at p = 0 (12.5) and 12.7 px against 11.2 px at
    p = 0.25 (9.8). Inconclusive, with opposite directions at the two p.
- **2026-11-17, verdict for (c).** Switches per step in play.
  - Q-learning, requested category: 0.417 against 0.383 at p = 0 (0.063), weak
    evidence for; 0.446 against 0.370 at p = 0.25 (0.036), supported. Paddle
    direction: 0.321 against 0.288 (0.050) and 0.316 against 0.246 (0.031), the
    same pattern. Inconclusive overall, because the sheet requires both p.
  - PPO: requested 0.361 against 0.382 and 0.343 against 0.360; paddle
    direction 0.333 against 0.318 and 0.308 against 0.302. Inconclusive.
- **2026-11-17, policy change.** On the in-play observations shared by the B
  agents, A-trained and B-trained Q-learning agents request a different
  movement category in 61% of visits, two seeds from the same setting in 48%.
  For PPO the shares are 45% and 44%.

## P4

- **2026-11-16, before training.** The 300-step success in the rationale is
  hypothetical. With γ = 0.99 and c = 0.001, a success beats a 27-step loss at
  any length below 267 steps, and the tracking rule reaches 2 bricks in 63–93
  steps in B. The prediction and its test are unchanged.

- **2026-11-17, verdict.** Setting C against Setting B, final evaluation.
  - (a) Median length of failed episodes: Q-learning 50.9 against 53.8 steps
    (9.9), PPO 50.2 against 51.6 steps (7.0). Inconclusive for both: weak
    evidence against.
  - (b) PPO success: 87.8% in B, 93.7% in C (12.6). Inconclusive: weak evidence against.
  - (c) Q-learning success: 75.9% in B, 79.1% in C (19.4). Inconclusive, as the
    sheet predicted. The test is weak: any change below 19 points gives this class.
  - (d) Median length of successful episodes, ratio C/B: 0.978 for Q-learning
    and 0.976 for PPO. Contradicted for both (threshold 0.95).

## P5

- **2026-11-17, verdict.** Training episodes from the final 10% of the budget
  against the final evaluation.
  - Order: the agents tie on training return and on evaluation success in A
    and B. In C PPO leads on training return (0.837 against 0.754) and the
    evaluation is a tie. Inconclusive: the ties prevent a comparison of orders.
  - Evaluation minus training success: A, Q-learning +3.8 and PPO +1.5 points
    (3.7), weak evidence for. B, −5.2 and +2.1 (12.5), weak evidence against.
    C, −3.4 and +3.0 (14.2), weak evidence against. In B and C the greedy
    Q-learning policy scores below its own training episodes.

## P6

- **2026-11-17, conditions for changing the initial recommendation (Setting B).**
  1. Not met: Q-learning's failure rate is 20.8%, PPO's 11.5%.
  2. Met: PPO's worst seed reaches 39.0%, Q-learning's worst 46.0%.
  3. Not met: 500,000 steps take 353 s for Q-learning and 383 s for PPO. At
     353 s Q-learning is at 76.1% and PPO at 82.6% (18.9).
  4. Not met: P7 is not contradicted.

## P7

- **2026-11-17, verdict.** Final B agents with the 16 px and the 12 px paddle.
  - Q-learning: 75.9% against 53.4% (13.5). Supported.
  - PPO: 87.8% against 68.9% (19.6). Inconclusive: weak evidence for.
  - Difference: Q-learning loses 22.5 points, PPO 18.9 (12.2). Inconclusive:
    weak evidence for.

## S1

- **2026-11-16, before training.** The rationale assumed that the start moves
  leave the paddle near the centre. Measured: the paddle starts at 99 on a
  range of 8–144, and more than a quarter of the starts end at the right wall.
  The common launch bins are therefore on the right, not in the centre. The
  prediction is unchanged: "common" and "rare" bins are identified from the
  logged visit counts.

- **2026-11-17, verdict.** The right-wall bin holds 32.6% of the starts and
  every other bin 5.8–11.7%. "Common" is a bin with at least 1/8 of the starts,
  which gives the right wall only. The definition was fixed in `src/probes.py`
  before the probe was run.
  - Q-learning, FIRE in common bins: FIRE alone has the highest value at the
    right wall in 7 of 10 seeds. Supported.
  - Q-learning, later launch from rare bins: step 6.8 against 2.4 (4.9).
    Inconclusive: weak evidence for. With the four most common bins counted as
    common (checked afterwards), 4.9 against 2.4 (1.8) would be supported.
  - PPO: FIRE is the most probable action in 79 of 80 (seed, bin) pairs, but
    P(FIRE) ≤ 0.9 in 50 of them (lowest 0.34). Contradicted.
- **2026-11-17, change of interpretation.** The delay does not come from tied
  values: no launch cell has four equal values. Each Q-learning agent has one
  launch position. Seven drive the paddle to the right wall and fire there, one
  uses the left wall and two a bin right of centre. In the other bins FIRE is
  the best action in at most one seed.

## S2

- **2026-11-17, verdict.** 112 shared S2 cells with 11,391 visits.
  - Q-learning favours STATIONARY in 43% of the visits (above half in 3 of 10
    seeds) and a move towards the ball in 34%. Contradicted.
  - PPO favours a move towards the ball in 30% (above half in no seed) and
    STATIONARY in 61%. Contradicted.
  - The observed difference is the reverse of the predicted one.
  - Contact offset: Q-learning 2.9 px, PPO 3.3 px (0.8). Inconclusive: weak
    evidence against wider Q-learning offsets.

## S3

- **2026-11-17, verdict.** 280 shared S3 cells.
  - Switches between consecutive steps inside S3: Q-learning 0.201, PPO 0.348
    (0.134). Contradicted: PPO switches more often.
  - Change of the favoured category between consecutive checkpoints:
    Q-learning 35.2%, PPO 24.7% (5.1). Supported. Over the last five
    transitions only (checked afterwards): 24.6% against 13.8% (5.4).
  - Agreement of at least 80% of the seeds: PPO 57.9% of the visits,
    Q-learning 6.3%. Supported.
  - PPO moves towards the ball in 64% of the visits where the ball is more
    than 4 px from the paddle centre (above half in 8 of 10 seeds). Supported.
