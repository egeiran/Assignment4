# Team review record (Part 5)

**Evidence reviewed:** the submitted prediction sheet (`61bcabe`), its dated
annotations, the evidence brief and saved analyses. Eivind supplied written
answers and a review on 24 November 2026 (`4cd8bf7`). This consolidation is
based on those answers and Trym's three written judgements supplied on
28 November. Their positions are attributed below; no additional meeting or
unrecorded approval is implied.

## 1. Strongest supporting evidence

Eivind identifies the ten-seed result in Setting A. Q-learning succeeded in
all 2,000 evaluation episodes (100.0 ± 0.0% across ten trained agents); PPO
reached 99.4 ± 2.1%, with nine seeds at 100% and one at 93.5%. He considers
this the strongest support for a classroom demonstration and evidence that
this environment, representation and final budget permit successful learning.
It does not separate the algorithms or establish performance in B–D.
Trym independently selects the ten perfect Q-learning agents as the strongest
finding and regards this as grounds for confidence in a classroom demo in A.

Eivind also notes that Q-learning seeds 0 and 6 reached 0% at some late checkpoints:
the final result hides sensitivity to when training stops. The independent
training replications are the ten agents, not 2,000 independent training runs;
A's 200 evaluation seeds average about 33 distinct trajectories per agent.

## 2. Claims that went beyond the evidence

Eivind challenges a broad reading of P2 as showing greater PPO seed variation.
His short answer accepts A's formal classification, whereas his longer review
proposes saying P2 is not supported because A's entire spread comes from one
PPO agent and B is inconclusive. The interpretation concern is distinguished
from the original classification. The technical check confirms:
A is formally supported under the explicit SD threshold (2.1 ≥ 1.5 × 0),
while B is inconclusive (17.5/16.7 = 1.05). Greater PPO variability in both
settings is not established. No new formal classification is introduced.
Trym explicitly retains A's preregistered classification and narrows the
conclusion: he would tone down the claim that PPO generally varies more
between training runs. This preserves the rule while addressing Eivind's
concern about the strength of the interpretation.

Eivind also challenges the classroom explainability argument. Q-values can be
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
Trym likewise interprets the shrinking gap as sensitivity to the chosen
settings. His proposed next experiment gives both methods the same tuning
budget, then compares the selected configurations on fresh training seeds.
This is future work, not a new experiment claimed in this submission.

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
Trym would try PPO first if the specific trained agent can be tested on fresh
episodes before use. Without such testing, he considers the evidence too weak
to recommend either method unconditionally when failures matter. This is a
conditional preference, not evidence that screening has already been validated.

Eivind's seed-6 GIF observation is retained as supplemental evidence. Existing
step logs corroborate a catch followed by stationary FIRE requests in one
episode; neither source explains all 122 failures. The original ten episodes
are unchanged.

## What the review changed or retained

The technical revision retains the original verdict rules and selected
configurations; qualifies explainability, generalisation and timing claims;
withdraws unconditional PPO deployment; and separates S2 and seed-6
observations from causal explanations. Trym's answers retain Q-learning for a
demo in A and make PPO the first candidate to test in B, while withholding an
unconditional recommendation. The review keeps A's formal P2 verdict with its
single-run limitation rather than adopting the broader "P2 is not supported"
wording from Eivind's longer review. Equal-budget retuning and fresh-seed
assessment remain proposed follow-up work. No new training was required or run.
