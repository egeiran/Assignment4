| Agent | Outcome | Episodes | Share | Most common length range | Episodes in the range | Median in the range |
| --- | --- | --- | --- | --- | --- | --- |
| Q-learning | success | 1518 | 75.9% | 70–79 | 781 | 74 |
| Q-learning | failure | 416 | 20.8% | 50–59 | 234 | 55 |
| Q-learning | timeout | 66 | 3.3% | – | – | – |
| PPO | success | 1756 | 87.8% | 70–79 | 1052 | 71 |
| PPO | failure | 230 | 11.5% | 50–59 | 162 | 53 |
| PPO | timeout | 14 | 0.7% | – | – | – |

| ID | Kind | Agent | Training seed | Evaluation seed | Outcome | Steps | Selected because | Source run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TS1 | typical success | Q-learning | 4 | 10005 | success | 74 | 74 steps; most common range 70–79 steps holds 781 of 1518 Q-learning successes, median 74 | results/runs/main/q_B_selected_s4 |
| TS2 | typical success | PPO | 1 | 10000 | success | 71 | 71 steps; most common range 70–79 steps holds 1052 of 1756 PPO successes, median 71 | results/runs/main/ppo_B_selected_s1 |
| TS3 | typical success | PPO | 3 | 10001 | success | 71 | 71 steps; most common range 70–79 steps holds 1052 of 1756 PPO successes, median 71 | results/runs/main/ppo_B_selected_s3 |
| TF1 | typical failure | Q-learning | 4 | 10003 | failure | 55 | 55 steps; most common range 50–59 steps holds 234 of 416 Q-learning failures, median 55 | results/runs/main/q_B_selected_s4 |
| TF2 | typical failure | PPO | 9 | 10000 | failure | 53 | 53 steps; most common range 50–59 steps holds 162 of 230 PPO failures, median 53 | results/runs/main/ppo_B_selected_s9 |
| TF3 | typical failure | Q-learning | 1 | 10020 | failure | 55 | 55 steps; most common range 50–59 steps holds 234 of 416 Q-learning failures, median 55 | results/runs/main/q_B_selected_s1 |
| SS1 | surprising success | Q-learning | 2 | 10077 | success | 96 | value estimate at launch 0.356, -2.41 SD from the mean of this agent's episodes (0.451) | results/runs/main/q_B_selected_s2 |
| SS2 | surprising success | PPO | 9 | 10107 | success | 72 | value estimate at launch 0.463, -5.29 SD from the mean of this agent's episodes (0.508) | results/runs/main/ppo_B_selected_s9 |
| SF1 | surprising failure | Q-learning | 1 | 10172 | failure | 53 | mean value estimate over the last five steps 0.411; median over Q-learning failures 0.152 | results/runs/main/q_B_selected_s1 |
| SF2 | surprising failure | PPO | 9 | 10004 | failure | 71 | mean value estimate over the last five steps 0.664; median over PPO failures 0.448 | results/runs/main/ppo_B_selected_s9 |

| Agent | Training seed | Successes | Failures | Timeouts | Selected episodes |
| --- | --- | --- | --- | --- | --- |
| PPO | 0 | 185 | 15 | 0 | – |
| PPO | 1 | 194 | 6 | 0 | TS2 |
| PPO | 2 | 194 | 6 | 0 | – |
| PPO | 3 | 187 | 13 | 0 | TS3 |
| PPO | 4 | 183 | 17 | 0 | – |
| PPO | 5 | 191 | 9 | 0 | – |
| PPO | 6 | 78 | 122 | 0 | – |
| PPO | 7 | 187 | 13 | 0 | – |
| PPO | 8 | 171 | 15 | 14 | – |
| PPO | 9 | 186 | 14 | 0 | TF2, SS2, SF2 |
| Q-learning | 0 | 108 | 92 | 0 | – |
| Q-learning | 1 | 92 | 42 | 66 | TF3, SF1 |
| Q-learning | 2 | 183 | 17 | 0 | SS1 |
| Q-learning | 3 | 131 | 69 | 0 | – |
| Q-learning | 4 | 158 | 42 | 0 | TS1, TF1 |
| Q-learning | 5 | 161 | 39 | 0 | – |
| Q-learning | 6 | 190 | 10 | 0 | – |
| Q-learning | 7 | 140 | 60 | 0 | – |
| Q-learning | 8 | 182 | 18 | 0 | – |
| Q-learning | 9 | 173 | 27 | 0 | – |
