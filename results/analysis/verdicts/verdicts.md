| ID | Part | Quantity | Result | Verdict | Evidence |
| --- | --- | --- | --- | --- | --- |
| P1 | Setting A | Success at 10% of the budget (PPO higher) | PPO 16.6%, Q-learning 17.4%; difference -0.9, pooled SD 25.5 | inconclusive | no measurable difference |
| P1 | Setting B | Success at 10% of the budget (PPO higher) | PPO 21.6%, Q-learning 11.1%; difference +10.5, pooled SD 23.8 | inconclusive | weak evidence for |
| P2 | Setting A | Between-seed SD of final success (PPO at least 1.5 x Q-learning) | PPO 2.1, Q-learning 0.0; ratio undefined (Q-learning SD is 0) | supported | supports |
| P2 | Setting B | Between-seed SD of final success (PPO at least 1.5 x Q-learning) | PPO 17.5, Q-learning 16.7; ratio 1.05 | inconclusive | no measurable difference |
| P3a | Q-learning | Final success, A-trained in A against B-trained in B (A higher) | A 100.0%, B 75.9%; difference +24.1, pooled SD 11.8 | supported | supports |
| P3a | PPO | Final success, A-trained in A against B-trained in B (A higher) | A 99.4%, B 87.8%; difference +11.6, pooled SD 12.4 | inconclusive | weak evidence for |
| P3b | Q-learning, p = 0 | Mean distance between ball x and paddle centre in descent at y ≥ 150, px (smaller for B-trained) | A-trained 11.50, B-trained 18.27; difference -6.77, pooled SD 6.08 | contradicted | contradicts |
| P3b | Q-learning, p = 0.25 | Mean distance between ball x and paddle centre in descent at y ≥ 150, px (smaller for B-trained) | A-trained 23.78, B-trained 21.86; difference +1.91, pooled SD 5.40 | inconclusive | weak evidence for |
| P3b | Q-learning, both p | Predicted direction required at both p | contradicted / inconclusive | contradicted | contradicts |
| P3b | PPO, p = 0 | Mean distance between ball x and paddle centre in descent at y ≥ 150, px (smaller for B-trained) | A-trained 9.14, B-trained 11.84; difference -2.70, pooled SD 12.46 | inconclusive | weak evidence against |
| P3b | PPO, p = 0.25 | Mean distance between ball x and paddle centre in descent at y ≥ 150, px (smaller for B-trained) | A-trained 12.71, B-trained 11.24; difference +1.47, pooled SD 9.82 | inconclusive | weak evidence for |
| P3b | PPO, both p | Predicted direction required at both p | inconclusive / inconclusive | inconclusive | mixed: weak evidence against / weak evidence for |
| P3c requested | Q-learning, p = 0 | Requested movement-category switches per step in play (fewer for B-trained) | A-trained 0.417, B-trained 0.383; difference +0.034, pooled SD 0.063 | inconclusive | weak evidence for |
| P3c requested | Q-learning, p = 0.25 | Requested movement-category switches per step in play (fewer for B-trained) | A-trained 0.446, B-trained 0.370; difference +0.076, pooled SD 0.036 | supported | supports |
| P3c requested | Q-learning, both p | Predicted direction required at both p | inconclusive / supported | inconclusive | mixed: weak evidence for / supports |
| P3c requested | PPO, p = 0 | Requested movement-category switches per step in play (fewer for B-trained) | A-trained 0.361, B-trained 0.382; difference -0.021, pooled SD 0.096 | inconclusive | weak evidence against |
| P3c requested | PPO, p = 0.25 | Requested movement-category switches per step in play (fewer for B-trained) | A-trained 0.343, B-trained 0.360; difference -0.016, pooled SD 0.088 | inconclusive | weak evidence against |
| P3c requested | PPO, both p | Predicted direction required at both p | inconclusive / inconclusive | inconclusive | weak evidence against |
| P3c observed | Q-learning, p = 0 | Paddle-direction switches per step in play (fewer for B-trained) | A-trained 0.321, B-trained 0.288; difference +0.033, pooled SD 0.050 | inconclusive | weak evidence for |
| P3c observed | Q-learning, p = 0.25 | Paddle-direction switches per step in play (fewer for B-trained) | A-trained 0.316, B-trained 0.246; difference +0.070, pooled SD 0.031 | supported | supports |
| P3c observed | Q-learning, both p | Predicted direction required at both p | inconclusive / supported | inconclusive | mixed: weak evidence for / supports |
| P3c observed | PPO, p = 0 | Paddle-direction switches per step in play (fewer for B-trained) | A-trained 0.333, B-trained 0.318; difference +0.015, pooled SD 0.084 | inconclusive | weak evidence for |
| P3c observed | PPO, p = 0.25 | Paddle-direction switches per step in play (fewer for B-trained) | A-trained 0.308, B-trained 0.302; difference +0.006, pooled SD 0.071 | inconclusive | no measurable difference |
| P3c observed | PPO, both p | Predicted direction required at both p | inconclusive / inconclusive | inconclusive | mixed: weak evidence for / no measurable difference |
| P3 policy change | Q-learning | Share of shared in-play observations (1642 cells) with a different requested category | A-trained against B-trained 0.609; two seeds of the same setting 0.481; pooled SD 0.036 | descriptive | – |
| P3 policy change | PPO | Share of shared in-play observations (1642 cells) with a different requested category | A-trained against B-trained 0.450; two seeds of the same setting 0.437; pooled SD 0.131 | descriptive | – |
| P4a | Q-learning | Median length of failed episodes (shorter in C) | B 50.9 steps, C 53.8 steps; difference -2.9, pooled SD 9.9 | inconclusive | weak evidence against |
| P4a | PPO | Median length of failed episodes (shorter in C) | B 50.2 steps, C 51.6 steps; difference -1.4, pooled SD 7.0 | inconclusive | weak evidence against |
| P4b | PPO | Final success (lower in C) | B 87.8%, C 93.7%; difference -5.9, pooled SD 12.6 | inconclusive | weak evidence against |
| P4c | Q-learning | Final success (change within the pooled SD) | B 75.9%, C 79.1%; difference +3.2, pooled SD 19.4 | inconclusive, as predicted | no clear change, as predicted; the test is weak |
| P4d | Q-learning | Median length of successful episodes, ratio C/B (at most 0.95) | B 73.8 steps, C 72.2 steps; ratio 0.978 | contradicted | contradicts |
| P4d | PPO | Median length of successful episodes, ratio C/B (at most 0.95) | B 72.1 steps, C 70.4 steps; ratio 0.976 | contradicted | contradicts |
| P5 order | Setting A | Leader by training return and by evaluation success | training return: PPO 0.979, Q-learning 0.962 (tie); evaluation success: tie | tie | – |
| P5 order | Setting B | Leader by training return and by evaluation success | training return: PPO 0.857, Q-learning 0.811 (tie); evaluation success: tie | tie | – |
| P5 order | Setting C | Leader by training return and by evaluation success | training return: PPO 0.837, Q-learning 0.754 (PPO); evaluation success: tie | tie | – |
| P5 order | overall | Orders differ in at least one setting | A: tie / tie; B: tie / tie; C: PPO / tie | inconclusive | ties prevent a comparison |
| P5 gap | Setting A | Evaluation minus training success (larger for Q-learning) | Q-learning 3.8 points, PPO 1.5 points; difference +2.3, pooled SD 3.7 | inconclusive | weak evidence for |
| P5 gap | Setting B | Evaluation minus training success (larger for Q-learning) | Q-learning -5.2 points, PPO 2.1 points; difference -7.3, pooled SD 12.5 | inconclusive | weak evidence against |
| P5 gap | Setting C | Evaluation minus training success (larger for Q-learning) | Q-learning -3.4 points, PPO 3.0 points; difference -6.4, pooled SD 14.2 | inconclusive | weak evidence against |
| P7 drop | Q-learning | Final success, 16 px against 12 px paddle (lower with 12 px) | 16 px 75.9%, 12 px 53.4%; difference +22.5, pooled SD 13.5 | supported | supports |
| P7 drop | PPO | Final success, 16 px against 12 px paddle (lower with 12 px) | 16 px 87.8%, 12 px 68.9%; difference +18.9, pooled SD 19.6 | inconclusive | weak evidence for |
| P7 difference | both agents | Drop in percentage points (smaller for PPO) | Q-learning 22.5 points, PPO 18.9 points; difference +3.6, pooled SD 12.2 | inconclusive | weak evidence for |
| D | Q-learning | Final success, original against gap layout | original 75.9%, gap 51.5%; difference -24.5, pooled SD 16.6 | no prediction | – |
| D | PPO | Final success, original against gap layout | original 87.8%, gap 83.4%; difference -4.4, pooled SD 23.2 | no prediction | – |
| P6.1 | Setting B | Q-learning's failure rate lower than PPO's by more than the pooled SD | Q-learning 20.8%, PPO 11.5%; pooled SD 15.4 | not triggered | – |
| P6.2 | Setting B | PPO's worst seed worse than Q-learning's worst seed | worst PPO seed 39.0%, worst Q-learning seed 46.0% | triggered | – |
| P6.3 | Setting B | Q-learning ahead at equal training time (353 s, interpolated) | time for 500,000 steps: Q-learning 353 s, PPO 383 s; success at 353 s: Q-learning 76.1%, PPO 82.6%; pooled SD 18.9 | not triggered | – |
| P6.4 | P7 | P7 contradicted | P7 difference is inconclusive | not triggered | – |
| S1 Q-learning | launch delay | Median launch step, rare against common launch bins (later in rare bins) | rare 6.8, common 2.4; difference +4.4, pooled SD 4.9; common bins: [7] | inconclusive | weak evidence for |
| S1 Q-learning | launch delay, four most common bins | Median launch step, rare against common launch bins (later in rare bins) | rare 4.9, common 2.4; difference +2.5, pooled SD 1.8; common bins: [0, 4, 6, 7] | sensitivity check | supports |
| S1 Q-learning | FIRE in common bins | Share of (seed, common bin) pairs where FIRE alone has the highest Q-value | 70% of 10 pairs; all four values tied in 0 pairs | supported | supports |
| S1 PPO | all bins | P(FIRE) > 0.9 in every launch bin | P(FIRE) ≤ 0.9 in 50 of 80 (seed, bin) pairs; lowest 0.344; FIRE most probable in 79 pairs | contradicted | contradicts |
| S2 Q-learning | action | Share of shared S2 observations (112 cells, 11391 visits) with STATIONARY favoured | mean 0.431 (SD 0.085); above half in 3 of 10 seeds; towards the ball 0.344; ties 0.028 | contradicted | contradicts |
| S2 PPO | action | Share of shared S2 observations with a move towards the ball favoured | mean 0.297 (SD 0.062); above half in 0 of 10 seeds; stationary 0.609 | contradicted | contradicts |
| S2 contact | both agents | Mean distance between ball x and paddle centre at the last falling step before a bounce, px (wider for Q-learning) | Q-learning 2.92, PPO 3.30; difference -0.38, pooled SD 0.80 | inconclusive | weak evidence against |
| S3 switches | both agents | Requested-category switches per consecutive step pair inside S3 (higher for Q-learning) | Q-learning 0.201, PPO 0.348; difference -0.147, pooled SD 0.134 | contradicted | contradicts |
| S3 checkpoints | both agents | Share of shared S3 observations (280 cells) whose favoured category changes between consecutive checkpoints (higher for Q-learning) | Q-learning 0.352, PPO 0.247; difference +0.105, pooled SD 0.051 | supported | supports |
| S3 checkpoints | second half of training | The same share over the last five transitions only | Q-learning 0.246, PPO 0.138; difference +0.108, pooled SD 0.054 | sensitivity check | supports |
| S3 seed agreement | both agents | Share of shared S3 observations where at least 80% of seeds favour the same category (larger for PPO) | Q-learning 0.063, PPO 0.579; Q-learning ties 0.028 | supported | supports |
| S3 PPO | direction | Share of shared S3 observations (ball more than 4 px from the paddle centre) with a move towards the ball favoured | mean 0.643 (SD 0.220); above half in 8 of 10 seeds | supported | supports |
