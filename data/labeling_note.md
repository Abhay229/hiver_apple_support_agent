# Golden Evaluation Set — Sampling & Labeling Note

**Source:** 74,613 real "customer opens a thread → AppleSupport's first reply" pairs,
reconstructed from the Kaggle Customer Support on Twitter dataset (twcs.csv, 2.81M rows),
filtered to English-language threads.

**Sampling method:** A rule-based keyword classifier (priority-ordered regex per intent)
was run over all 74,613 messages to build 7 candidate pools matching our draft taxonomy.
We then stratified-sampled 210 messages, deliberately **over-sampling rare intents**
(e.g. warranty_repair, only ~1.3% of the corpus) rather than sampling purely proportional
to real prevalence — a proportional sample would leave only ~3 examples for rare classes,
too few to compute reliable per-class precision/recall later.

**Labeling method:** Every one of the 210 sampled messages was then read and manually
labeled/corrected by hand — the keyword classifier's pre-label was treated as a
first guess only, not ground truth. On review, ~40% of pre-labels were wrong, mostly
because generic terms ("screen", "update", "connect") match both real bugs and unrelated
complaints. 9 messages were dropped for being non-English (they slipped past a crude
ASCII-ratio filter). Final set: **201 examples.**

**Notable finding (kept honest, not smoothed over):** the keyword heuristic estimated
~15-20% of messages as "pure venting with no fixable detail." After manual reading, only
~2.5% (5/201) genuinely had zero actionable technical content — almost all angry-sounding
tweets do name a real, specific issue once read carefully. This means an intent classifier
tuned on keyword-based labels would badly overestimate how often the agent should punt to
"can't classify, escalate."

**Routing label:** each example also got an AUTO_HANDLE / ESCALATE label based on a stated
policy (not vibes): how-to questions and known bugs with a published public fix →
AUTO_HANDLE; anything needing account access, serial numbers, case IDs, or open-ended
diagnostic back-and-forth → ESCALATE. 44/201 (22%) were labeled AUTO_HANDLE.

**Final intent distribution (201 examples):**
| Intent | Count |
|---|---|
| software_bug_or_update | 112 |
| account_subscription | 25 |
| warranty_repair | 23 |
| how_to_question | 13 |
| connectivity_compat | 13 |
| hardware_defect | 10 |
| general_complaint_venting | 5 |
