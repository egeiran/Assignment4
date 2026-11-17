| ID | Part | Quantity | Result | Verdict | Evidence |
| --- | --- | --- | --- | --- | --- |
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
