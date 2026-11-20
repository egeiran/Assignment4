## TS1, typical success: Q-learning, training seed 4, evaluation seed 10005. Success after 74 steps

- Selected because: 74 steps; most common range 70–79 steps holds 781 of 1518 Q-learning successes, median 74.
- Source run: `results/runs/main/q_B_selected_s4`.
- From the first step, 77 of 100 replays with new sticky noise succeed.
- Largest fall in the success rate of the requested action (20 replays per step): after step 40, from 100% to 70%. The best action at step 40 reached 100%; the best action after the fall reaches 85%.

| Evidence | Decision 1, step 8 |
| --- | --- |
| State and available actions | Ball at x 55, y 145, moving down and left; 64 px left of the paddle centre (x 119), 44 px above the paddle. Bricks hit: 0 of 2. State cell 1936. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and Q-values | NOOP. NOOP 0.378, FIRE 0.274, RIGHT 0.282, LEFT 0.276. Value estimate 0.378 |
| Observed transition and outcome | Paddle centre x 119 → 119; ball → x 51, y 149. Next brick hit at step 34. Episode: success at step 74 |
| Alternative action | Episode success in 100 replays per action: NOOP 89%, FIRE 81%, RIGHT 88%, LEFT 91%. Reading: near-equal actions |

| Evidence | Decision 2, step 17 |
| --- | --- |
| State and available actions | Ball at x 19, y 181, moving down and left; 3 px right of the paddle centre (x 16), 8 px above the paddle. Bricks hit: 0 of 2. State cell 3701. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and Q-values | LEFT. NOOP 0.438, FIRE 0.428, RIGHT 0.428, LEFT 0.451. Value estimate 0.451 |
| Observed transition and outcome | Paddle centre x 16 → 16; ball → x 15, y 185. Next brick hit at step 34. Episode: success at step 74 |
| Alternative action | Episode success in 100 replays per action: NOOP 94%, FIRE 90%, RIGHT 5%, LEFT 90%. Reading: the action matters; the chosen action is among the best |

| Evidence | Decision 3, step 48 |
| --- | --- |
| State and available actions | Ball at x 115, y 178, moving down and right; 1 px left of the paddle centre (x 116), 11 px above the paddle. Bricks hit: 1 of 2. State cell 4152. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and Q-values | FIRE. NOOP 0.482, FIRE 0.698, RIGHT 0.484, LEFT 0.592. Value estimate 0.698 |
| Observed transition and outcome | Paddle centre x 116 → 135; ball → x 119, y 184. Next brick hit at step 74. Episode: success at step 74 |
| Alternative action | Episode success in 100 replays per action: NOOP 98%, FIRE 95%, RIGHT 0%, LEFT 73%. Reading: the action matters; the chosen action is among the best |

## TS2, typical success: PPO, training seed 1, evaluation seed 10000. Success after 71 steps

- Selected because: 71 steps; most common range 70–79 steps holds 1052 of 1756 PPO successes, median 71.
- Source run: `results/runs/main/ppo_B_selected_s1`.
- From the first step, 98 of 100 replays with new sticky noise succeed.
- Largest fall in the success rate of the requested action (20 replays per step): after step 41, from 95% to 80%. The best action at step 41 reached 100%; the best action after the fall reaches 100%.

| Evidence | Decision 1, step 3 |
| --- | --- |
| State and available actions | Ball at x 67, y 133, moving down and left; 62 px left of the paddle centre (x 129), 56 px above the paddle. Bricks hit: 0 of 2. State cell 1936. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and action probabilities | LEFT. NOOP 0.027, FIRE 0.006, RIGHT 0.000, LEFT 0.966. Value estimate 0.484 |
| Observed transition and outcome | Paddle centre x 129 → 107; ball → x 63, y 137. Next brick hit at step 32. Episode: success at step 71 |
| Alternative action | Episode success in 100 replays per action: NOOP 100%, FIRE 95%, RIGHT 98%, LEFT 99%. Reading: near-equal actions |

| Evidence | Decision 2, step 24 |
| --- | --- |
| State and available actions | Ball at x 27, y 139, moving up and right; 11 px right of the paddle centre (x 16), 50 px above the paddle. Bricks hit: 0 of 2. State cell 2185. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and action probabilities | FIRE. NOOP 0.079, FIRE 0.623, RIGHT 0.297, LEFT 0.001. Value estimate 0.611 |
| Observed transition and outcome | Paddle centre x 16 → 16; ball → x 31, y 133. Next brick hit at step 32. Episode: success at step 71 |
| Alternative action | Episode success in 100 replays per action: NOOP 98%, FIRE 98%, RIGHT 97%, LEFT 99%. Reading: near-equal actions |

| Evidence | Decision 3, step 46 |
| --- | --- |
| State and available actions | Ball at x 115, y 178, moving down and right; 1 px left of the paddle centre (x 116), 11 px above the paddle. Bricks hit: 1 of 2. State cell 4152. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and action probabilities | FIRE. NOOP 0.066, FIRE 0.928, RIGHT 0.006, LEFT 0.000. Value estimate 0.698 |
| Observed transition and outcome | Paddle centre x 116 → 127; ball → x 119, y 184. Next brick hit at step 71. Episode: success at step 71 |
| Alternative action | Episode success in 100 replays per action: NOOP 91%, FIRE 95%, RIGHT 72%, LEFT 30%. Reading: the action matters; the chosen action is among the best |

## TS3, typical success: PPO, training seed 3, evaluation seed 10001. Success after 71 steps

- Selected because: 71 steps; most common range 70–79 steps holds 1052 of 1756 PPO successes, median 71.
- Source run: `results/runs/main/ppo_B_selected_s3`.
- From the first step, 99 of 100 replays with new sticky noise succeed.
- Largest fall in the success rate of the requested action (20 replays per step): after step 11, from 100% to 90%. The best action at step 11 reached 100%; the best action after the fall reaches 100%.

| Evidence | Decision 1, step 6 |
| --- | --- |
| State and available actions | Ball at x 55, y 145, moving down and left; 5 px right of the paddle centre (x 50), 44 px above the paddle. Bricks hit: 0 of 2. State cell 1920. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and action probabilities | FIRE. NOOP 0.277, FIRE 0.593, RIGHT 0.001, LEFT 0.130. Value estimate 0.532 |
| Observed transition and outcome | Paddle centre x 50 → 50; ball → x 51, y 149. Next brick hit at step 32. Episode: success at step 71 |
| Alternative action | Episode success in 100 replays per action: NOOP 97%, FIRE 98%, RIGHT 94%, LEFT 93%. Reading: near-equal actions |

| Evidence | Decision 2, step 38 |
| --- | --- |
| State and available actions | Ball at x 83, y 130, moving down and right; 2 px right of the paddle centre (x 81), 59 px above the paddle. Bricks hit: 1 of 2. State cell 2359. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and action probabilities | NOOP. NOOP 0.497, FIRE 0.446, RIGHT 0.000, LEFT 0.056. Value estimate 0.664 |
| Observed transition and outcome | Paddle centre x 81 → 81; ball → x 87, y 136. Next brick hit at step 71. Episode: success at step 71 |
| Alternative action | Episode success in 100 replays per action: NOOP 90%, FIRE 95%, RIGHT 85%, LEFT 93%. Reading: near-equal actions |

| Evidence | Decision 3, step 46 |
| --- | --- |
| State and available actions | Ball at x 115, y 178, moving down and right; level with the paddle centre (x 115), 11 px above the paddle. Bricks hit: 1 of 2. State cell 4152. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and action probabilities | FIRE. NOOP 0.455, FIRE 0.511, RIGHT 0.031, LEFT 0.002. Value estimate 0.752 |
| Observed transition and outcome | Paddle centre x 115 → 115; ball → x 119, y 184. Next brick hit at step 71. Episode: success at step 71 |
| Alternative action | Episode success in 100 replays per action: NOOP 100%, FIRE 100%, RIGHT 16%, LEFT 8%. Reading: the action matters; the chosen action is among the best |

## TF1, typical failure: Q-learning, training seed 4, evaluation seed 10003. Failure after 55 steps

- Selected because: 55 steps; most common range 50–59 steps holds 234 of 416 Q-learning failures, median 55.
- Source run: `results/runs/main/q_B_selected_s4`.
- From the first step, 72 of 100 replays with new sticky noise succeed.
- Largest fall in the success rate of the requested action (20 replays per step): after step 47, from 80% to 0%. The best action at step 47 reached 100%; the best action after the fall reaches 100%.

| Evidence | Decision 1, step 6 |
| --- | --- |
| State and available actions | Ball at x 63, y 137, moving down and left; 56 px left of the paddle centre (x 119), 52 px above the paddle. Bricks hit: 0 of 2. State cell 1936. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and Q-values | NOOP. NOOP 0.378, FIRE 0.274, RIGHT 0.282, LEFT 0.276. Value estimate 0.378 |
| Observed transition and outcome | Paddle centre x 119 → 119; ball → x 59, y 141. Next brick hit at step 34. Episode: failure at step 55 |
| Alternative action | Episode success in 100 replays per action: NOOP 86%, FIRE 85%, RIGHT 87%, LEFT 91%. Reading: near-equal actions |

| Evidence | Decision 2, step 17 |
| --- | --- |
| State and available actions | Ball at x 19, y 181, moving down and left; 3 px right of the paddle centre (x 16), 8 px above the paddle. Bricks hit: 0 of 2. State cell 3701. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and Q-values | LEFT. NOOP 0.438, FIRE 0.428, RIGHT 0.428, LEFT 0.451. Value estimate 0.451 |
| Observed transition and outcome | Paddle centre x 16 → 16; ball → x 15, y 185. Next brick hit at step 34. Episode: failure at step 55 |
| Alternative action | Episode success in 100 replays per action: NOOP 90%, FIRE 86%, RIGHT 8%, LEFT 90%. Reading: the action matters; the chosen action is among the best |

| Evidence | Decision 3, step 48 |
| --- | --- |
| State and available actions | Ball at x 115, y 178, moving down and right; 9 px right of the paddle centre (x 106), 11 px above the paddle. Bricks hit: 1 of 2. State cell 4154. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and Q-values | RIGHT. NOOP 0.277, FIRE 0.299, RIGHT 0.673, LEFT 0.272. Value estimate 0.673 |
| Observed transition and outcome | Paddle centre x 106 → 127; ball → x 119, y 184. Episode: failure at step 55 |
| Alternative action | Episode success in 100 replays per action: NOOP 99%, FIRE 100%, RIGHT 0%, LEFT 17%. Reading: poor action: FIRE succeeds 100 points more often |

## TF2, typical failure: PPO, training seed 9, evaluation seed 10000. Failure after 53 steps

- Selected because: 53 steps; most common range 50–59 steps holds 162 of 230 PPO failures, median 53.
- Source run: `results/runs/main/ppo_B_selected_s9`.
- From the first step, 96 of 100 replays with new sticky noise succeed.
- Largest fall in the success rate of the requested action (20 replays per step): after step 47, from 100% to 0%. The best action at step 47 reached 100%; the best action after the fall reaches 0%.

| Evidence | Decision 1, step 0 |
| --- | --- |
| State and available actions | Ball not launched. Paddle centre at x 152. Bricks hit: 0 of 2. State cell 4759. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and action probabilities | FIRE. NOOP 0.030, FIRE 0.929, RIGHT 0.003, LEFT 0.037. Value estimate 0.481 |
| Observed transition and outcome | Paddle centre x 152 → 152; ball → x 75, y 125. Next brick hit at step 32. Episode: failure at step 53 |
| Alternative action | Episode success in 100 replays per action: NOOP 96%, FIRE 95%, RIGHT 97%, LEFT 96%. Reading: near-equal actions |

| Evidence | Decision 2, step 35 |
| --- | --- |
| State and available actions | Ball at x 71, y 112, moving down and right; 4 px left of the paddle centre (x 75), 77 px above the paddle. Bricks hit: 1 of 2. State cell 1765. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and action probabilities | NOOP. NOOP 0.531, FIRE 0.185, RIGHT 0.283, LEFT 0.001. Value estimate 0.725 |
| Observed transition and outcome | Paddle centre x 75 → 75; ball → x 75, y 118. Episode: failure at step 53 |
| Alternative action | Episode success in 100 replays per action: NOOP 96%, FIRE 95%, RIGHT 96%, LEFT 97%. Reading: near-equal actions |

| Evidence | Decision 3, step 46 |
| --- | --- |
| State and available actions | Ball at x 115, y 178, moving down and right; 6 px right of the paddle centre (x 109), 11 px above the paddle. Bricks hit: 1 of 2. State cell 4153. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and action probabilities | NOOP. NOOP 0.513, FIRE 0.358, RIGHT 0.128, LEFT 0.001. Value estimate 0.813 |
| Observed transition and outcome | Paddle centre x 109 → 109; ball → x 119, y 184. Episode: failure at step 53 |
| Alternative action | Episode success in 100 replays per action: NOOP 98%, FIRE 97%, RIGHT 100%, LEFT 0%. Reading: the action matters; the chosen action is among the best |

## TF3, typical failure: Q-learning, training seed 1, evaluation seed 10020. Failure after 55 steps

- Selected because: 55 steps; most common range 50–59 steps holds 234 of 416 Q-learning failures, median 55.
- Source run: `results/runs/main/q_B_selected_s1`.
- From the first step, 69 of 100 replays with new sticky noise succeed.
- Largest fall in the success rate of the requested action (20 replays per step): after step 8, from 90% to 40%. The best action at step 8 reached 90%; the best action after the fall reaches 75%.

| Evidence | Decision 1, step 17 |
| --- | --- |
| State and available actions | Ball at x 19, y 181, moving down and left; 3 px right of the paddle centre (x 16), 8 px above the paddle. Bricks hit: 0 of 2. State cell 3701. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and Q-values | FIRE. NOOP 0.563, FIRE 0.598, RIGHT 0.504, LEFT 0.573. Value estimate 0.598 |
| Observed transition and outcome | Paddle centre x 16 → 16; ball → x 15, y 185. Next brick hit at step 34. Episode: failure at step 55 |
| Alternative action | Episode success in 100 replays per action: NOOP 68%, FIRE 58%, RIGHT 6%, LEFT 69%. Reading: the action matters; the chosen action is among the best |

| Evidence | Decision 2, step 34 |
| --- | --- |
| State and available actions | Ball at x 59, y 94, moving up and right; 40 px left of the paddle centre (x 99), 95 px above the paddle. Bricks hit: 1 of 2. State cell 1034. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and Q-values | RIGHT. NOOP 0.496, FIRE 0.472, RIGHT 0.611, LEFT 0.487. Value estimate 0.611 |
| Observed transition and outcome | Paddle centre x 99 → 122; ball → x 63, y 100. Episode: failure at step 55 |
| Alternative action | Episode success in 100 replays per action: NOOP 97%, FIRE 98%, RIGHT 0%, LEFT 97%. Reading: poor action: FIRE succeeds 98 points more often |

| Evidence | Decision 3, step 42 |
| --- | --- |
| State and available actions | Ball at x 91, y 142, moving down and right; 61 px left of the paddle centre (x 152), 47 px above the paddle. Bricks hit: 1 of 2. State cell 2365. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and Q-values | FIRE. NOOP 0.575, FIRE 0.586, RIGHT 0.584, LEFT 0.573. Value estimate 0.586 |
| Observed transition and outcome | Paddle centre x 152 → 152; ball → x 95, y 148. Episode: failure at step 55 |
| Alternative action | Episode success in 100 replays per action: NOOP 0%, FIRE 0%, RIGHT 0%, LEFT 98%. Reading: poor action: LEFT succeeds 98 points more often |

## SS1, surprising success: Q-learning, training seed 2, evaluation seed 10077. Success after 96 steps

- Selected because: value estimate at launch 0.356, -2.41 SD from the mean of this agent's episodes (0.451).
- Source run: `results/runs/main/q_B_selected_s2`.
- From the first step, 93 of 100 replays with new sticky noise succeed.
- Largest fall in the success rate of the requested action (20 replays per step): after step 5, from 100% to 20%. The best action at step 5 reached 100%; the best action after the fall reaches 25%.

| Evidence | Decision 1, step 22 |
| --- | --- |
| State and available actions | Ball at x 77, y 187, moving down and left; 9 px right of the paddle centre (x 68), 2 px above the paddle. Bricks hit: 0 of 2. State cell 3714. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and Q-values | LEFT. NOOP 0.000, FIRE 0.000, RIGHT 0.001, LEFT 0.013. Value estimate 0.013 |
| Observed transition and outcome | Paddle centre x 68 → 60; ball → x 78, y 187. Next brick hit at step 47. Episode: success at step 96 |
| Alternative action | Episode success in 100 replays per action: NOOP 70%, FIRE 74%, RIGHT 72%, LEFT 30%. Reading: poor action: FIRE succeeds 44 points more often |

| Evidence | Decision 2, step 49 |
| --- | --- |
| State and available actions | Ball at x 69, y 103, moving down and left; 10 px right of the paddle centre (x 59), 86 px above the paddle. Bricks hit: 1 of 2. State cell 777. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and Q-values | LEFT. NOOP 0.000, FIRE 0.000, RIGHT 0.000, LEFT 0.000. Value estimate 0.000 |
| Observed transition and outcome | Paddle centre x 59 → 82; ball → x 63, y 107. Next brick hit at step 96. Episode: success at step 96 |
| Alternative action | Episode success in 100 replays per action: NOOP 68%, FIRE 66%, RIGHT 65%, LEFT 75%. Reading: near-equal actions |

| Evidence | Decision 3, step 70 |
| --- | --- |
| State and available actions | Ball at x 72, y 187, moving down and right; 19 px right of the paddle centre (x 53), 2 px above the paddle. Bricks hit: 1 of 2. State cell 4134. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and Q-values | RIGHT. NOOP 0.000, FIRE 0.000, RIGHT 0.043, LEFT 0.000. Value estimate 0.043 |
| Observed transition and outcome | Paddle centre x 53 → 76; ball → x 78, y 189. Next brick hit at step 96. Episode: success at step 96 |
| Alternative action | Episode success in 100 replays per action: NOOP 4%, FIRE 0%, RIGHT 100%, LEFT 12%. Reading: the action matters; the chosen action is among the best |

## SS2, surprising success: PPO, training seed 9, evaluation seed 10107. Success after 72 steps

- Selected because: value estimate at launch 0.463, -5.29 SD from the mean of this agent's episodes (0.508).
- Source run: `results/runs/main/ppo_B_selected_s9`.
- From the first step, 96 of 100 replays with new sticky noise succeed.
- Largest fall in the success rate of the requested action (20 replays per step): after step 1, from 100% to 90%. The best action at step 1 reached 100%; the best action after the fall reaches 90%.

| Evidence | Decision 1, step 7 |
| --- | --- |
| State and available actions | Ball at x 117, y 147, moving down and left; 5 px left of the paddle centre (x 122), 42 px above the paddle. Bricks hit: 0 of 2. State cell 1940. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and action probabilities | LEFT. NOOP 0.034, FIRE 0.323, RIGHT 0.000, LEFT 0.643. Value estimate 0.470 |
| Observed transition and outcome | Paddle centre x 122 → 110; ball → x 113, y 151. Next brick hit at step 33. Episode: success at step 72 |
| Alternative action | Episode success in 100 replays per action: NOOP 97%, FIRE 93%, RIGHT 77%, LEFT 91%. Reading: near-equal actions |

| Evidence | Decision 2, step 16 |
| --- | --- |
| State and available actions | Ball at x 81, y 183, moving down and left; 6 px left of the paddle centre (x 87), 6 px above the paddle. Bricks hit: 0 of 2. State cell 3711. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and action probabilities | LEFT. NOOP 0.013, FIRE 0.102, RIGHT 0.000, LEFT 0.885. Value estimate 0.513 |
| Observed transition and outcome | Paddle centre x 87 → 75; ball → x 81, y 184. Next brick hit at step 33. Episode: success at step 72 |
| Alternative action | Episode success in 100 replays per action: NOOP 90%, FIRE 93%, RIGHT 7%, LEFT 98%. Reading: the action matters; the chosen action is among the best |

| Evidence | Decision 3, step 47 |
| --- | --- |
| State and available actions | Ball at x 53, y 181, moving down and right; 8 px right of the paddle centre (x 45), 8 px above the paddle. Bricks hit: 1 of 2. State cell 4132. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and action probabilities | RIGHT. NOOP 0.101, FIRE 0.009, RIGHT 0.890, LEFT 0.000. Value estimate 0.722 |
| Observed transition and outcome | Paddle centre x 45 → 58; ball → x 57, y 185. Next brick hit at step 72. Episode: success at step 72 |
| Alternative action | Episode success in 100 replays per action: NOOP 81%, FIRE 77%, RIGHT 100%, LEFT 0%. Reading: the action matters; the chosen action is among the best |

## SF1, surprising failure: Q-learning, training seed 1, evaluation seed 10172. Failure after 53 steps

- Selected because: mean value estimate over the last five steps 0.411; median over Q-learning failures 0.152.
- Source run: `results/runs/main/q_B_selected_s1`.
- From the first step, 65 of 100 replays with new sticky noise succeed.
- Largest fall in the success rate of the requested action (20 replays per step): after step 29, from 70% to 10%. The best action at step 29 reached 100%; the best action after the fall reaches 100%.

| Evidence | Decision 1, step 15 |
| --- | --- |
| State and available actions | Ball at x 19, y 181, moving down and left; 3 px right of the paddle centre (x 16), 8 px above the paddle. Bricks hit: 0 of 2. State cell 3701. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and Q-values | FIRE. NOOP 0.563, FIRE 0.598, RIGHT 0.504, LEFT 0.573. Value estimate 0.598 |
| Observed transition and outcome | Paddle centre x 16 → 16; ball → x 15, y 185. Next brick hit at step 32. Episode: failure at step 53 |
| Alternative action | Episode success in 100 replays per action: NOOP 72%, FIRE 67%, RIGHT 3%, LEFT 58%. Reading: the action matters; the chosen action is among the best |

| Evidence | Decision 2, step 32 |
| --- | --- |
| State and available actions | Ball at x 59, y 94, moving up and right; 40 px left of the paddle centre (x 99), 95 px above the paddle. Bricks hit: 1 of 2. State cell 1034. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and Q-values | RIGHT. NOOP 0.496, FIRE 0.472, RIGHT 0.611, LEFT 0.487. Value estimate 0.611 |
| Observed transition and outcome | Paddle centre x 99 → 122; ball → x 63, y 100. Episode: failure at step 53 |
| Alternative action | Episode success in 100 replays per action: NOOP 98%, FIRE 97%, RIGHT 0%, LEFT 98%. Reading: poor action: NOOP succeeds 98 points more often |

| Evidence | Decision 3, step 40 |
| --- | --- |
| State and available actions | Ball at x 91, y 142, moving down and right; 61 px left of the paddle centre (x 152), 47 px above the paddle. Bricks hit: 1 of 2. State cell 2365. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and Q-values | FIRE. NOOP 0.575, FIRE 0.586, RIGHT 0.584, LEFT 0.573. Value estimate 0.586 |
| Observed transition and outcome | Paddle centre x 152 → 152; ball → x 95, y 148. Episode: failure at step 53 |
| Alternative action | Episode success in 100 replays per action: NOOP 0%, FIRE 0%, RIGHT 0%, LEFT 98%. Reading: poor action: LEFT succeeds 98 points more often |

## SF2, surprising failure: PPO, training seed 9, evaluation seed 10004. Failure after 71 steps

- Selected because: mean value estimate over the last five steps 0.664; median over PPO failures 0.448.
- Source run: `results/runs/main/ppo_B_selected_s9`.
- From the first step, 97 of 100 replays with new sticky noise succeed.
- Largest fall in the success rate of the requested action (20 replays per step): after step 63, from 100% to 0%. The best action at step 63 reached 100%; the best action after the fall reaches 0%.

| Evidence | Decision 1, step 2 |
| --- | --- |
| State and available actions | Ball at x 86, y 128, moving down and right; 61 px right of the paddle centre (x 25), 61 px above the paddle. Bricks hit: 0 of 2. State cell 1726. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and action probabilities | RIGHT. NOOP 0.001, FIRE 0.000, RIGHT 0.999, LEFT 0.000. Value estimate 0.404 |
| Observed transition and outcome | Paddle centre x 25 → 46; ball → x 90, y 132. Next brick hit at step 40. Episode: failure at step 71 |
| Alternative action | Episode success in 100 replays per action: NOOP 92%, FIRE 94%, RIGHT 93%, LEFT 91%. Reading: near-equal actions |

| Evidence | Decision 2, step 16 |
| --- | --- |
| State and available actions | Ball at x 142, y 184, moving down and right; 5 px right of the paddle centre (x 137), 5 px above the paddle. Bricks hit: 0 of 2. State cell 4120. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and action probabilities | RIGHT. NOOP 0.155, FIRE 0.089, RIGHT 0.755, LEFT 0.001. Value estimate 0.496 |
| Observed transition and outcome | Paddle centre x 137 → 140; ball → x 147, y 184. Next brick hit at step 40. Episode: failure at step 71 |
| Alternative action | Episode success in 100 replays per action: NOOP 100%, FIRE 100%, RIGHT 100%, LEFT 21%. Reading: the action matters; the chosen action is among the best |

| Evidence | Decision 3, step 62 |
| --- | --- |
| State and available actions | Ball at x 129, y 182, moving down and right; 2 px right of the paddle centre (x 127), 7 px above the paddle. Bricks hit: 1 of 2. State cell 4152. Actions: NOOP, FIRE, RIGHT, LEFT |
| Chosen action and action probabilities | FIRE. NOOP 0.380, FIRE 0.616, RIGHT 0.003, LEFT 0.001. Value estimate 0.818 |
| Observed transition and outcome | Paddle centre x 127 → 127; ball → x 135, y 186. Episode: failure at step 71 |
| Alternative action | Episode success in 100 replays per action: NOOP 99%, FIRE 98%, RIGHT 100%, LEFT 6%. Reading: the action matters; the chosen action is among the best |
