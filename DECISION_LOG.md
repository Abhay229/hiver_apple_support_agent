# Decision Log

Non-obvious decisions made during this project, in roughly chronological order.

1. **Didn't fake the dataset when Kaggle/HuggingFace were network-blocked.**
   Found a GitHub mirror first, but it only had 100 rows — too small to be
   real. Asked the user to upload the actual `twcs.csv` rather than proceed
   on a synthetic or tiny stand-in, since the whole point of the assignment
   is handling real, messy data.

2. **Picked AppleSupport over the higher-volume AmazonHelp.** AmazonHelp had
   more raw tweets, but its replies were almost entirely generic secure-link
   redirects with no real resolution content to ground on, and heavily
   multilingual. AppleSupport had a genuine mix of real public
   troubleshooting and account-specific escalation — better material for
   both reply-grounding and the routing decision.

3. **Derived the intent taxonomy from reading real data, not from Banking77
   or an assumed list.** Read a random sample of messages first, then
   validated category proportions with a keyword pass before finalizing 7
   intents — kept deliberately small since the assignment asked for "a
   small set."

4. **Golden-set sampling deliberately over-samples rare intents** rather
   than sampling proportional to real prevalence — a proportional sample
   would leave ~3 examples for `warranty_repair` (~1.3% of the corpus), too
   few for a trustworthy per-class metric. Traded distributional realism
   for statistical usability, and said so explicitly.

5. **Treated the keyword pre-classifier as a candidate generator only, never
   ground truth.** Manually re-read and corrected all 210 sampled examples
   by hand; ~40% of pre-labels were wrong. This also revealed the true
   `general_complaint_venting` rate was ~2.5%, not the ~15-20% the keyword
   heuristic suggested — kept the honest smaller number rather than pad the
   category to its original target size.

6. **Dropped 9 non-English messages** that slipped through a crude
   ASCII-ratio filter (Spanish/Italian/Portuguese/Dutch), caught only by
   manually reading every sampled example.

7. **Classify only the opening message of a thread, not any mid-thread
   turn.** Required extra work to reconstruct true thread-starters, because
   that's what a real incoming-ticket classifier actually sees — unlike
   most rows in the raw dataset.

8. **Wrote the AUTO_HANDLE/ESCALATE policy as an explicit, auditable rule**
   (which clause justifies each decision) rather than an implicit judgment
   call baked into a prompt — makes routing inspectable and consistent
   instead of vibes-based.

9. **Chose TF-IDF retrieval over neural embeddings for reply grounding.**
   Weaker on pure paraphrase matching, but free, deterministic, and easy to
   debug when a retrieved example looks wrong — a stated tradeoff, not an
   assumed-optimal choice.

10. **Actively excluded all golden-set examples from the retrieval corpus**
    (exact text match) before any evaluation, so a prediction can never be
    grounded on retrieving its own ground-truth answer — easy to miss, and
    would have quietly inflated every downstream number.

11. **Kept the simple baseline's keyword classifier exactly as first
    drafted, not cleaned up.** A fair "simple baseline" should reflect what
    a naive first pass actually looks like, known ~40% error rate included
    — not a secretly-improved version.

12. **Wrote the simple baseline's routing rule independently from the
    policy used to build the gold labels**, specifically to avoid a
    circular baseline that would trivially "solve" routing by reusing the
    exact logic that generated the answer key.

13. **Made the judge's safety check a hard gate, not a soft score.** A
    failed `safety_pass` zeroes the whole composite reply-quality score
    instead of being averaged in as just another 1-5 dimension — a
    well-worded but unsafe reply should score badly, not "pretty good."

14. **Refused to fabricate live-agent numbers when sandbox API access
    wasn't available.** Marked those report rows as pending with exact
    instructions to fill them in, rather than inventing a plausible-looking
    accuracy or quality score.

15. **Declined a live API key pasted into chat, and instead hand-scored 30
    real replies myself for the judge-agreement check** — the harder but
    safer path on both counts. The manual scoring is also what actually
    surfaced several of the report's failure modes (nearest-neighbor
    mismatches, tone-deafness on a distressing hardware complaint).

16. **Reported real numbers even when they looked bad, instead of chasing a
    prettier result.** The live agent's 40.3% intent accuracy is worse than
    both baselines at face value. Rather than keep swapping models/providers
    until a nicer top-line number appeared, reported it as-is and diagnosed
    *why* — 52% of calls failed to return parseable JSON on the free-tier
    model, which is the real story, not the raw accuracy number.

17. **Documented the LLM-judge pipeline's failure rather than faking or
    omitting reply-quality results.** The free-tier model used for judging
    also failed to produce valid JSON almost every call, collapsing
    composite scores to ~0 and breaking the agreement calculation on NaNs.
    Chose to report this honestly as a real limitation (with the
    already-completed 30-example hand-scored reference standing in as the
    actual reply-quality evidence) rather than hide the gap or invent
    numbers to fill the table.