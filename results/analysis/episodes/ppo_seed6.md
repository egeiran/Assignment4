# Supplemental look at PPO seed 6 in Setting B

This is an additional descriptive analysis of an existing run. It does
not replace or alter the prespecified ten-episode Part 4 selection.
The same 200 evaluation seeds (10,000–10,199) are used at every
checkpoint; policies are greedy and evaluation does not update them.

## Evaluation checkpoints

| Steps | Success | Failure | Timeout |
| ---: | ---: | ---: | ---: |
| 0 | 0.0% | 0.0% | 100.0% |
| 25,000 | 0.0% | 76.0% | 24.0% |
| 50,000 | 20.5% | 79.5% | 0.0% |
| 100,000 | 14.5% | 85.5% | 0.0% |
| 150,000 | 11.0% | 51.5% | 37.5% |
| 200,000 | 16.0% | 68.0% | 16.0% |
| 250,000 | 11.0% | 89.0% | 0.0% |
| 300,000 | 45.5% | 54.5% | 0.0% |
| 350,000 | 14.5% | 85.5% | 0.0% |
| 400,000 | 47.0% | 53.0% | 0.0% |
| 450,000 | 21.0% | 79.0% | 0.0% |
| 500,000 | 39.0% | 61.0% | 0.0% |

## Final-policy episodes

At 500,000 steps, 122 episodes fail and 0 time out. Of the failures, 121 launch at step 1 and 1 at step 2. Failure length has median 53 steps (range 25–74); success length has median 82 (range 79–88).
The failures occur after launch, but these episode rows do not include per-step states for the unselected failures and do not identify why the ball was lost.

## Supplemental GIF checked against saved per-step probes

The GIF `results/videos/ppo_B_selected_s6_B_seed10001.gif` shows one 53-step failure. Its requested actions agree with the saved probe for evaluation seed 10001. The ball is falling at step 16 and rising at step 17 after a paddle contact. From decision step 20 through 52, the paddle stays at RAM-derived x = 8 (the encoder's centre is 16) and the requested action is FIRE. The first brick is hit before the ball returns and is lost. These are observations of this trajectory, not evidence that FIRE or a particular learning mechanism explains all 122 failures. No alternative-action replay was added.

## Training record and limits

The final 50,000-interaction window contains 846 completed
episodes: 232 successes and 614 failures
(27.4% of completed episodes succeeded).
Across the full run, 1,686 of 9,076
completed episodes succeeded (18.6%).
The 500,000 training interactions took 399.2 s;
scheduled evaluations took 696.3 s.

These records describe the run and its non-monotone checkpoint scores.
They do not identify a cause for the low success rate, establish
convergence, or estimate how often similar runs occur. The episodes
were not selected through the ten-episode annotation rule, and no new
alternative-action replay was run for this seed.

**Sources:** `results/runs/main/ppo_B_selected_s6/eval_episodes.csv.gz`,
`eval_checkpoints.csv`, `train_episodes.csv`, `runtime.json`.
Per-step corroboration: `results/runs/probes/ppo_B_selected_s6_on_B/` (`steps.npz`, `eval_episodes.csv.gz`, `eval_checkpoints.csv`).
