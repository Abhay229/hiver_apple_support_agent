# Report — AppleSupport AI Agent

## 1. Problem framing

**What "good" means for this brand.** AppleSupport's brand is built on polish
and trust, and the real data shows their agents mostly do two things well:
(a) give a genuinely helpful, grounded public answer when one is safe to give
publicly, and (b) pull the conversation into DM the moment it needs account
access, a serial number, or a real diagnostic back-and-forth. So "good" here
is **not** "maximize how many messages get auto-answered." A wrong public
reply is worse than an unnecessary escalation. Concretely, "good" means, in
priority order:

1. **Never send an unsafe or fabricated reply** (a made-up URL, a promise
   Apple can't keep, a request for sensitive info in public). This is a hard
   gate in the judge (`safety_pass`), not a soft score.
2. **Escalate when genuinely uncertain**, rather than optimize for a high
   auto-handle rate. A false ESCALATE (human handles something simple) costs
   a little time. A false AUTO_HANDLE (agent answers something it shouldn't
   have) can actively damage trust.
3. **Ground replies in how Apple actually resolves the issue**, not
   plausible-sounding generic help-desk text.
4. **Classify intent accurately enough to route correctly** — perfect
   end-to-end resolution in one turn is not the bar; correct routing is.

**What I chose not to build**, and why:

- **No fine-tuning.** Prompting + retrieval over real historical resolutions
  gets most of the value at a fraction of the engineering cost, and stays
  auditable (you can see exactly which historical examples grounded a
  reply). Fine-tuning was out of scope for a one-brand take-home.
- **No multi-turn conversation modeling.** The agent only sees the opening
  customer message, mirroring how a triage system would first encounter a
  ticket. Real Apple resolutions often continue in DM, which this dataset
  can't see — a real limitation, discussed in §4.
- **No image/media understanding.** Several real messages reference an
  attached screenshot ("look at my screen... notice the time"). The agent
  works from text only; these are systematically harder cases.
- **No real language detection.** English-only, filtered with a crude
  ASCII-ratio heuristic rather than a real language-ID model. Known
  limitation (§3, failure mode 5).
- **No automatic account verification/linking.** Anything needing account
  access is escalated, full stop — building a fake authentication flow was
  out of scope and arguably unsafe to fake.
- **No live Twitter posting/production pipeline.** This is a classify +
  draft + route pipeline, not a deployed bot.

## 2. Results vs. baselines

Two baselines, both computed offline against the 201-example golden set
(see `data/baseline_summary.csv` for the run):

| Approach | Intent accuracy | Routing accuracy |
|---|---|---|
| **Trivial** (always guess `software_bug_or_update`, always `ESCALATE`) | 55.7% | 78.1% |
| **Simple** (first-draft keyword regex + copy nearest-neighbor historical reply) | 63.2% | 26.9% |
| **LLM agent** (retrieval-grounded, this system) | *[fill in after `run_eval.py` + your API key]* | *[fill in]* |
| **LLM agent reply quality** (judge composite, /5) | *[fill in after `judge.py`]* | — |

I'm reporting the agent's own numbers as pending rather than fabricating
them: I have no live model access in the environment I built this in (see
decision log). The harness (`eval_harness.py`) is wired to produce this row
the moment it's run with a real key — that's the actual reproducibility
test for this repo.

The one result I can already discuss honestly: **the simple baseline is
worse than the trivial one on routing**, despite "looking" smarter. Its
keyword rule (escalate only if a sensitive word like "password" or "warranty"
appears) misses the large class of messages that need escalation for a
different reason — an open-ended diagnostic conversation — and confidently
mislabels them AUTO_HANDLE. This is exactly the trap a real team could fall
into shipping a keyword MVP: better-sounding logic, worse actual safety
behavior.

## 3. Failure analysis — top 5 failure modes

All five are drawn from real examples surfaced while hand-scoring 30 replies
against the rubric (`data/human_judge_scores.csv`), not hypothetical ones.

**1. Retrieval mismatch on vague or short messages.**
Example: *"can you tell me where Facebook & Twitter has gone in my settings...
need to put password in for Facebook"* (G129) retrieved a phishing-reporting
article as its nearest historical neighbor — plausible lexical overlap
("password") but the wrong topic entirely.
*Hypothesis:* short, keyword-sparse messages don't have enough distinctive
vocabulary for TF-IDF similarity to disambiguate; it latches onto whichever
salient word happens to co-occur with the retrieved historical example,
regardless of topical fit.

**2. "Popular answer" bias / canned-line overfitting.**
Example: the exact same *"Here's what you can do to work around the issue
until it's fixed in a future software update"* + link (Apple's real,
widely-reused reply to the famous iOS 11.1 autocorrect bug) got retrieved
for battery/WiFi complaints (G173) and a home-button/media-key bug (G051)
that are unrelated to autocorrect.
*Hypothesis:* because that one bug was so viral, its reply appears dozens of
times in the corpus with generic-sounding customer phrasing ("bug," "since
update," "annoying"), so it dominates retrieval for any vaguely-worded
post-update complaint — a specific instance of popularity bias in
nearest-neighbor retrieval.

**3. Tone-deafness on severe or emotionally distressed complaints.**
Example: *"my phone is just dying before my eyes... screen is broken and
floated, front camera does not work... my eyes hurt, I'm tired... help me
please 😭😭😭"* (G121, a serious hardware failure with real emotional
content) retrieved a reply about how to adjust screen brightness.
*Hypothesis:* keyword/retrieval systems have no signal for emotional
severity or urgency, so nothing in the pipeline up-weights "this needs a
careful, human, empathetic response" over "this superficially matches a
settings question."

**4. Multi-intent / compound messages lose the secondary ask.**
Example: several messages vent at length and then ask a genuine, separate
question buried at the end (e.g. G161: profanity-laden preamble, then "how
can I block numbers"). A single-label classifier picks one bucket and the
concrete, answerable question can get lost behind the dominant emotional
framing.
*Hypothesis:* forcing one intent label per message is a real modeling
simplification that costs accuracy on compound messages — a meaningful
minority of real support messages aren't single-intent.

**5. Non-English messages slipping past a crude language filter.**
9 of the 210 originally-sampled golden candidates were Spanish, Italian,
Portuguese, or Dutch despite passing an ASCII-ratio filter (accented
European text is still mostly ASCII). I caught these by manually reading
every example; a live system relying only on this heuristic would send
English-templated replies to non-English speakers.
*Hypothesis:* ASCII-ratio is a poor proxy for "is this English" for any
Latin-script language; needs a real language-ID model (e.g. `langdetect`
or `fastText`) rather than a string heuristic.

## 4. What is misleading about my headline number?

This section is mandatory and I'm taking it seriously rather than
box-ticking it, because most of what looked like a clean result in this
project had a real caveat underneath once I checked.

- **A 78% routing accuracy sounds decent, but it's just majority-class
  guessing.** The trivial baseline gets this "for free" by always predicting
  ESCALATE, because 78% of the golden set genuinely is ESCALATE. Its recall
  on the minority AUTO_HANDLE class is 0% — it is never right about the
  case that matters most for measuring whether auto-handling is safe. Any single blended accuracy number on an imbalanced label hides this;
  per-class recall (not shown by the top-line number alone) is the number
  that actually matters here.
- **A single "intent accuracy" number averages over wildly different
  class-level difficulty.** 55.7% of the golden set is one class
  (`software_bug_or_update`). Any classifier that's decent at spotting that
  one dominant class looks good on blended accuracy while potentially doing
  badly on `hardware_defect` or `warranty_repair` (n=10 and n=23
  respectively) — too few examples in this golden set to report those
  per-class numbers with much confidence in the first place.
- **The apparent prevalence of "pure venting with no fixable detail" was off
  by 6-8x before manual review** (an early keyword-based estimate said
  ~15-20%; manual reading of every example found ~2.5%). This isn't just an
  isolated data quirk — it's a warning about *any* number in this report
  that traces back to a keyword heuristic rather than an actual read. I've
  tried to flag which numbers are heuristic-derived vs. hand-verified
  throughout, but the honest takeaway is: trust the manually-verified 201,
  not any of the earlier keyword tallies.
- **The golden set has exactly one labeler (me), one pass, no
  inter-annotator agreement on the labels themselves.** I do measure
  judge-vs-human agreement (§ eval harness), but that only checks whether
  the LLM judge agrees with *my* reply-quality scores — it says nothing
  about whether a second human would have labeled the *intents* the same
  way I did. A systematic blind spot in my own labeling (e.g. how I resolved
  ambiguous update-caused-data-loss cases) would silently bias every
  downstream number.
- **Contamination risk on this specific dataset.** The Customer Support on
  Twitter dataset, and the iOS 11.1 autocorrect bug specifically, are
  extremely well-documented and were almost certainly in the training data
  of any capable LLM used here. A high reply-quality score on messages about
  that bug may partly reflect the model already knowing the famous fix,
  not a generalizable support-agent skill that would transfer to a private
  company's undocumented internal issues. This is a real limit on how far
  these results generalize.
- **Judge and agent may share a model family.** If the same model (or a
  close relative) both drafts a reply and grades it, self-preference bias is
  a real risk that would inflate the reported composite score. I have not
  controlled for this in the current setup.
- **"First reply quality" is not "issue resolved."** Every ground-truth
  reply in this dataset is Apple's *first* public response, and a large
  fraction of those are themselves just "please DM us" with the actual fix
  happening in a private, unobserved channel. So even a perfect score
  against this golden set measures "did the agent's first turn look like a
  good first turn," not "did the customer's problem actually get fixed."

## 5. What I'd do with one more week

1. **Run the live agent + judge on the full 201-example golden set** — the
   one concrete thing still missing from this report, and the top priority.
2. **Get a second human labeler for a 50-example subset** of the golden set
   to compute real inter-annotator agreement on intent labels, not just
   judge-vs-human agreement on reply quality.
3. **Replace the ASCII-ratio language filter** with a real language-ID
   model, and decide a real policy for non-English messages (translate?
   route to a language-specific queue? always escalate?).
4. **Test for judge self-preference bias** by scoring the same replies with
   a judge from a different model family and checking whether scores shift.
5. **Build a genuinely held-out contamination check** — evaluate on Apple
   support tweets from after any plausible training cutoff, or from a
   less-documented, smaller brand, to see whether reply quality holds up
   without the benefit of the model already knowing the famous bugs in this
   specific dataset.
6. **Allow multi-label intent tagging** for compound messages instead of
   forcing a single label, and measure how often that actually changes the
   routing decision.
7. **Expand the golden set specifically for the rare classes** (only 10
   hardware_defect and 13 how_to_question examples currently) so per-class
   metrics are trustworthy, not just the blended top-line number.
