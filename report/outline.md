# Outline for the evidence brief

Working document, not a deliverable. The brief may have at most 2,000 words,
excluding tables and figure captions. Numbers therefore go in tables and the
text explains what they mean. The word count is checked with a script before
submission.

## Sections and word budget

| # | Section | Words | Answers | Tables and figures |
| - | ------- | ----- | ------- | ------------------ |
| 1 | Setup and comparison rules | 200 | Part 2 rules 1–8: budget and why, seeds, evaluation policy, tuning allowance, hardware | T1 setup. T2 tuning grid and selected configuration |
| 2 | Settings A–D | 400 | Part 2: efficiency (A), effect of uncertainty (B), incentives (C, outcome rates only), transfer (D) and whether the state can generalise | T3 required evidence per condition. F1 learning curves A, B, C. T4 transfer: B, D and P7 per agent |
| 3 | Mechanisms | 400 | Part 3: five investigations, what stays fixed; one prediction revisited, a case that does not fit, a competing explanation, the diagnostic experiment | T5 final success per value. F2 curves per investigation. T6 diagnostic result |
| 4 | Individual episodes | 300 | Part 4: how the ten episodes were selected, what they show that the tables hide, poor action or unlucky outcome | F3 episode-length distributions. T7 the ten episodes. T8 annotated decisions (three per episode) |
| 5 | Recommendations | 550 | Part 6: three situations, each with configuration, two pieces of evidence, one weakness, a change condition and one further experiment | T9 one row per situation |
| 6 | Scope | 150 | What is specific to this implementation and environment, what illustrates a broader principle. No claim that one algorithm is superior in general | – |
| | **Total** | **2,000** | | |

The prediction verdicts (P1–P7, S1–S3) are a table (T10) in
`predictions/annotations.md`, referenced from sections 2–4. The team review
(Part 5) is a separate document, `report/review_record.md`.

## Required evidence, per condition (T3)

One row per agent and setting, summarised across trained agents (seeds), never
pooled over episodes: success rate, failure rate, timeout rate, mean evaluation
return, median length of successful and of failed episodes, training
interactions, training time, between-seed SD and range. Returns in C are not
compared with A and B.

## Rules for every figure and table

- The caption names the source runs (`results/runs/<group>/...`) or the
  `sources.txt` that lists them.
- Curves: what is on the x-axis (environment interactions), how training curves
  are smoothed (10,000-step windows) and how often evaluation occurs.
- Evaluation policy stated: greedy Q with random tie-breaks, most probable PPO
  action; sampled PPO reported separately.

## Points that must not be lost

- Timeouts are a separate outcome. An agent whose greedy policy never launches
  the ball fails differently from one that loses the ball.
- In Setting A the evaluation episodes repeat; report distinct trajectories.
- The state holds no brick information. This limits what Setting D can show.
- PPO is evaluated at the policy of its last completed update.
- Hardware, and that runs shared the machine (8 processes), next to runtimes.
