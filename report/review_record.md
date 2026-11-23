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
2. **Configuration uncertainty:** Tuning selected Q-learning α = 0.3 on seeds
   100–102, while the separate matched seeds 0–4 favour α = 0.1 by 19.2 points.
   Should we keep the algorithm comparison explicitly conditional on the
   selected configurations, and do we want a matched ten-seed α follow-up
   before recommending an algorithm for B?
3. **Outlying PPO run:** Does seed 6's non-monotone evaluation (11% at 250k,
   39% at 500k) justify requiring a pre-deployment evaluation of each trained
   agent? What trace or replay would we need before explaining why it failed?
   The original ten selected episodes are unchanged; this is a separate
   descriptive addition, not a replacement selection.
4. **Diagnostic claim:** The forced action is followed by the agent policy to
   episode end; still-versus-towards success differs by 8.0 points (Q-learning)
   and 2.2 points (PPO), both below pooled SD. Do we agree this leaves the
   effect unresolved without proving practical equivalence, given that three
   Q-learning seeds lose 17–40 points after TOWARDS? Would a randomized S2
   state sample materially improve the inference?
5. **Generalisation and budget:** Is the revised claim appropriately narrow:
   the state carries useful ball/paddle geometry across layouts but omits
   brick positions and paddle width? Do we agree that solving A at 500k does
   not establish sufficient training in B or D?
6. **Predictions and recommendations:** After checking
   `results/analysis/verdicts/verdicts.md` against the submitted rules, do we
   accept its component-wise P1–P7 and S1–S3 classifications? Do the proposed
   recommendations distinguish training time from evaluation, tuning and
   inference cost, and do we want them changed?

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
