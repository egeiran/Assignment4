# Review record (Part 5)

Maximum 1,000 words. This is a preparation sheet for the required discussion
between Eivind and Trym; it is not a completed group review.

**Prepared:** 19 November 2026, by Codex (AI), from the evidence brief,
prediction annotations, analysis summaries, saved run records and source code.
**Human review and approval:** not yet completed. Neither person's answers or
agreement is inferred here. The submitted prediction sheet remains unchanged.

## Questions for Eivind and Trym

1. **Strongest evidence:** Is the matched ten-seed comparison in A (Q-learning
   100% on every seed; PPO 99.4% ± 2.1) the strongest result, or does the
   matched B comparison better support a useful recommendation despite its
   larger seed variation? Why?
   - Eivind: I think A is best for the classroom demo as Q-learning reaches 100% in all 10 trainingseeds, and PPO does it for 9/10. I would say both solving the deterministic setup with 500k steps har good support. In B PPO has a lower observed failrate, and higher success in 8/10 matched seeds. At the same time PPO has the weakest indiviudal run at 39% against Q-learnings 46%. For operation of a pretrained agent both of these observations can be important. So in short - A is best for the demo, and B for the suggestion about operation under uncertainty, even though the advantage of B is inconclusive according to our own rule. Looking at the different failures - 
2. **Configuration uncertainty:** Tuning selected Q-learning α = 0.3 on seeds
   100–102, while the separate matched seeds 0–4 favour α = 0.1 by 19.2 points.
   Should we keep the algorithm comparison explicitly conditional on the
   selected configurations, and do we want a matched ten-seed α follow-up
   before recommending an algorithm for B?
   - Eivind: I would say the algorithmcomparison should be explicitly conditional on the selected configs, and we should maybe run a new matched copmarison before giving a suggesting B further. The numbers given in the question above make learningrate a plausible explanation for some of PPOs head start. 
3. **Outlying PPO run:** Does seed 6's non-monotone evaluation (11% at 250k,
   39% at 500k) justify requiring a pre-deployment evaluation of each trained
   agent? What trace or replay would we need before explaining why it failed?
   The original ten selected episodes are unchanged; this is a separate
   descriptive addition, not a replacement selection.
   - Eivind: I would say that each agent should be pre-deployment evaluated if fails mean a lot, but I don't have really have too much to say on this one. Running `uv run python -B src/make_video.py results/runs/main/ppo_B_selected_s6 --seed 10001` shows that after the paddle hits the ball it just stays still and tries to FIRE
4. **Diagnostic claim:** The forced action is followed by the agent policy to
   episode end; still-versus-towards success differs by 8.0 points (Q-learning)
   and 2.2 points (PPO), both below pooled SD. Do we agree this leaves the
   effect unresolved without proving practical equivalence, given that three
   Q-learning seeds lose 17–40 points after TOWARDS? Would a randomized S2
   state sample materially improve the inference?
   - Eivind: The original prediction is opposite for both, but the consequence of the prediction is still open under our deciding rule as it is still over 0 (so it isn't denied either)
5. **Generalisation and budget:** Is the revised claim appropriately narrow:
   the state carries useful ball/paddle geometry across layouts but omits
   brick positions and paddle width? Do we agree that solving A at 500k does
   not establish sufficient training in B or D?
   - Eivind: I would say yes - we would most likely improve by increasing the number of runs. Especially considering that some of the B-runs are below their own in the end. 
6. **Predictions and recommendations:** After checking
   `results/analysis/verdicts/verdicts.md` against the submitted rules, do we
   accept its component-wise P1–P7 and S1–S3 classifications? Do the proposed
   recommendations distinguish training time from evaluation, tuning and
   inference cost, and do we want them changed?
   - Eivind: I would generally keep the verdicts and specified what each suggestion depends on. For example - P1 is unclear with the decided budget, P2 is supportet in A under the rule with one deviated PPO seed and so on. 

## Eivind review:
## 1. Strongest supporting evidence

The ten-seed result in Setting A. Q-learning succeeded in all 2,000 evaluation
episodes (100.0 ± 0.0% over 10 seeds); PPO reached 99.4 ± 2.1%, with nine seeds
at 100% and seed 6 at 93.5% (Table 2). It is strong because it holds for every
seed, rests on 200 episodes per agent, and both algorithms show the same. It
supports the classroom recommendation and shows that the environment, state and
budget work. It does not separate the algorithms or say anything about B–D, and
the final score hides that Q-learning seeds 0 and 6 succeeded in no evaluation
episode at one or two late checkpoints, so when training stops matters.

## 2. A claim that goes beyond the evidence

`verdicts.md` classifies P2 (PPO varies more between seeds) as supported in
Setting A. The rule is met (SD 2.1 against 0.0), but all the variation comes
from one PPO agent at 93.5%; the other 19 agents reach 100%. The prediction
covered both A and B, and in B it is inconclusive (SD 17.5 against 16.7).
Supported version: P2 is not supported; in A the only variation is one agent.

The brief says Q-learning's decisions "can be read from the table" (Table 7,
recommendation 1). Explainability was not measured, and `decisions.md` shows
PPO's action probabilities per decision just as plainly. Supported version:
both agents give per-state numbers that can be shown; that Q-values are easier
for a class to understand is an assumption.

## 3. An alternative explanation

Finding: PPO succeeds more often than Q-learning in B (87.8 against 75.9%).
Alternative: the gap reflects Q-learning's selected configuration (α 0.3,
chosen on three tuning seeds), not the algorithm. On seeds 0–4, α 0.1 reaches
86.4% and γ 0.9 86.1%, against 67.2% for the selected configuration. That
shrinks the gap to PPO (94.3%) from 27.1 to 7.9 points. With α 0.3, seeds 0
and 1 fall from 94 to 54% and from 92 to 46% in the last 100,000 steps
(Table 4). To tell the explanations apart, retune both agents on new seeds and
evaluate ten agents each on new evaluation seeds: a gap still above the pooled
SD points to the algorithm, a smaller one to the configuration. We cannot simply
switch to α 0.1, since that choice would use evaluation results (comparison
rule 7).

## 4. Comparison with the submitted predictions

Predictions about average differences between the algorithms were mostly
inconclusive: P1 (PPO learns faster) in A and B, P5 (training return predicts
evaluation) in every setting, and P4a–c (step penalty). P4d was contradicted:
successful episodes in C were 2% shorter, not 5%. Predictions about behaviour
in specific states were decided more often, in both directions: S2 was reversed
for both agents, S1 was contradicted for PPO, and three of four S3 parts were
supported. P3a and P7 were supported for Q-learning, which loses success under
sticky actions and with a narrower paddle; for PPO both were inconclusive.
P6's second condition was met (worst PPO seed 39% against 46%), so the initial
PPO recommendation no longer holds unconditionally. Setting D had no
prediction. 

## Human record (to complete after discussion)

- Date, attendees, and brief version reviewed:
- Strongest evidence and why:
- Claim revised or retained, with supported wording:
- Alternative explanation and observation that would distinguish it:
- Prediction verdicts changed, if any, with reason:
- Recommendation changes agreed by both:
- Resulting edits and commit:

No human judgement has been recorded yet. The evidence brief and annotations
contain AI-prepared analysis and proposed wording for the team to review; they
do not constitute the discussion or approval required for Part 5.
