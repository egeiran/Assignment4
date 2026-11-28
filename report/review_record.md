# Team review record (Part 5)

**Evidence reviewed:** the submitted prediction sheet (`61bcabe`), its dated
annotations, the evidence brief and saved analyses. Eivind supplied written
answers and a review on 24 November 2026 (`4cd8bf7`). This consolidation is
prepared on 28 November. Trym's answers to three follow-up questions are still
pending; no discussion outcome, agreement or approval is attributed to him.

## 1. Strongest supporting evidence

Eivind identifies the ten-seed result in Setting A. Q-learning succeeded in
all 2,000 evaluation episodes (100.0 ± 0.0% across ten trained agents); PPO
reached 99.4 ± 2.1%, with nine seeds at 100% and one at 93.5%. He considers
this the strongest support for a classroom demonstration and evidence that
this environment, representation and final budget permit successful learning.
It does not separate the algorithms or establish performance in B–D.

He also notes that Q-learning seeds 0 and 6 reached 0% at some late checkpoints:
the final result hides sensitivity to when training stops. The independent
training replications are the ten agents, not 2,000 independent training runs;
A's 200 evaluation seeds average about 33 distinct trajectories per agent.

## 2. Claims that went beyond the evidence

Eivind challenges a broad reading of P2 as showing greater PPO seed variation.
His short answer accepts A's formal classification, whereas his longer review
proposes saying P2 is not supported because A's entire spread comes from one
PPO agent and B is inconclusive. These passages need distinguishing rather
than silently replacing the original rule. The technical check confirms:
A is formally supported under the explicit SD threshold (2.1 ≥ 1.5 × 0),
while B is inconclusive (17.5/16.7 = 1.05). Greater PPO variability in both
settings is not established. No new formal classification is introduced.

He also challenges the classroom explainability argument. Q-values can be
inspected, but PPO's action probabilities are visible too. Neither student
understanding nor comparative explainability was measured. The brief now
states what a student can inspect and treats easier understanding as an
untested assumption.

## 3. A plausible alternative explanation

Eivind proposes that PPO's higher B success reflects the selected Q-learning
configuration rather than an intrinsic algorithm advantage. On the matched
training seeds 0–4, selected PPO scores 94.3%, selected Q-learning (α 0.3)
67.2%, and Q-learning with only α changed to 0.1 scores 86.4%. The PPO lead
shrinks from 27.1 to 7.9 percentage points. All five Q pairs favour α 0.1;
α 0.3 seeds 0 and 1 deteriorate in the last 100,000 steps.

He proposes retuning both agents on new tuning seeds, then evaluating new
trained agents on fresh evaluation seeds. The current comparison demonstrates
configuration sensitivity, not which algorithm is best tuned. Tuning seeds
100–102 favoured α 0.3 (83.7% versus 53.5%, with final ε 0.01 for both).
Choosing α 0.1 now using evaluation results would violate comparison rule 7.
A future study must keep selection separate from final assessment; crossing
the descriptive pooled-SD threshold alone would not establish a causal
explanation for a remaining gap.

## 4. Comparison with the original predictions

Eivind generally favours retaining the component-wise verdicts and stating
what each recommendation depends on. P1 is inconclusive at the specified
small budget; P5's ranking/gap predictions and P4a–c are mostly inconclusive.
P4d is contradicted: successful episodes shortened by about 2%, not 5%.
P3a and the narrower-paddle success drop support the Q-learning predictions,
while the corresponding PPO components are inconclusive. P7's predicted
between-agent difference is also inconclusive. D had no prior prediction.

The S2 majority-action predictions are contradicted for both agents; PPO's
S1 threshold is contradicted, and three of four S3 components are supported.
Eivind regards S2's consequences as unresolved. The technical distinction is
between preferred action, catch after a forced action and subsequent episode
success. Differences below pooled SD do not show practical equivalence;
Q-learning seeds 2, 6 and 8 lose 17–40 points after TOWARDS.

P6.2 was triggered: the worst observed PPO B agent scored 39%, versus 46% for
Q-learning. Eivind weighs this alongside PPO's higher B mean (87.8% versus
75.9%) and lower failure rate. He supports checking each agent before use
when failures matter. These minima are not guaranteed lower bounds, and no
acceptance threshold has been established.

His seed-6 GIF observation is retained as supplemental evidence. Existing
step logs corroborate a catch followed by stationary FIRE requests in one
episode; neither source explains all 122 failures. The original ten episodes
are unchanged.

## Changes and remaining decisions

The technical revision retains the original verdict rules and selected
configurations; qualifies explainability, generalisation and timing claims;
withdraws unconditional PPO deployment; and separates S2 and seed-6
observations from causal explanations. These edits do not constitute joint
human approval.

Trym's pending input concerns the B risk trade-off and recommendation, his own
strongest evidence and overclaim, and how configuration uncertainty should
shape the conclusion and next experiment. These answers are needed to finish
the team review, record agreement or dissent, and identify the final resulting
edits. No additional training has been agreed or run.
