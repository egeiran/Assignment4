# Annotations to the prediction sheet

The submitted sheet, `predictions_v1.md` / `.pdf` (merge commit `61bcabe`,
submitted 13 November 2026), is preserved unchanged. This file holds dated
notes keyed to its prediction IDs: changes of setup, verdicts under the sheet's
decision rule, and changes of interpretation. A note never replaces the
original prediction.

Verdicts use the sheet's rule: **supported**, **contradicted**, **inconclusive**
or **untested**, classified per component for multi-part predictions.

## Setup changes

- **2026-11-16, state size.** The state has 4,760 cells, not "about 2,100": a
  zero ball direction is kept as its own value (plan, section 2).
- **2026-11-16, open details fixed.** Bin boundaries, PPO implementation, seeds,
  tuning and evaluation points are fixed in `plans/part2_3_plan.md`, committed
  before any training.
- **2026-11-14, Setting D.** The team confirmed the sheet's 32 px gap layout as
  D. The sheet contains no prediction for this layout (P7 concerns the narrower
  paddle). D results are therefore reported without a prior prediction.

## P1

No annotations yet.

## P2

No annotations yet.

## P3

No annotations yet.

## P4

- **2026-11-16, before training.** The 300-step success in the rationale is
  hypothetical. With γ = 0.99 and c = 0.001, a success beats a 27-step loss at
  any length below 267 steps, and the tracking rule reaches 2 bricks in 63–93
  steps in B. The prediction and its test are unchanged.

## P5

No annotations yet.

## P6

No annotations yet.

## P7

No annotations yet.

## S1

- **2026-11-16, before training.** The rationale assumed that the start moves
  leave the paddle near the centre. Measured: the paddle starts at 99 on a
  range of 8–144, and more than a quarter of the starts end at the right wall.
  The common launch bins are therefore on the right, not in the centre. The
  prediction is unchanged: "common" and "rare" bins are identified from the
  logged visit counts.

## S2

No annotations yet.

## S3

No annotations yet.
