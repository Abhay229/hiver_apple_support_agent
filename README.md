# AppleSupport AI Agent — Hiver Take-Home

An AI support agent for `@AppleSupport`, built and evaluated on the real
Kaggle "Customer Support on Twitter" dataset. See `data/labeling_note.md`
for how the brand and intent taxonomy were chosen, and the full report
(coming next) for problem framing, baselines, and failure analysis.

## What's here

```
data/
  apple_opening_pairs_en.csv   74,613 real customer->AppleSupport opening
                                messages + replies, English-only (retrieval corpus)
  golden_eval_set.csv          201 hand-labeled examples (intent + routing)
  labeling_note.md             sampling & labeling methodology
src/
  intents.py                   the 7-intent taxonomy + routing policy (source of truth)
  retrieval.py                 TF-IDF retrieval over historical resolutions
  agent.py                     prompt construction + Anthropic API call + JSON parsing
  run_eval.py                  runs the agent over the golden set, reports accuracy
  build_corpus.py              regenerates the corpus from the raw Kaggle file (optional)
```

## Setup (2 minutes)

```bash
pip install -r requirements.txt
```

**LLM provider — free by default.** This pipeline runs on
[Groq](https://console.groq.com/keys) (free, no credit card, ~14,400
requests/day, plenty for this project) by default. Get a key in ~30 seconds
and set:

```bash
export GROQ_API_KEY=gsk_...
```

If you'd rather use a paid Anthropic key instead:

```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-...
```

Model choice is also overridable: `AGENT_MODEL` and `JUDGE_MODEL` env vars
(defaults: `llama-3.3-70b-versatile` on Groq, `claude-sonnet-5` on
Anthropic). Note: smaller/free open models can be a bit less reliable at
strict JSON formatting than Claude — `agent.py`/`judge.py` already handle a
JSON parse failure gracefully (marked `PARSE_ERROR` rather than crashing),
but if you see a lot of these, retry with a larger Groq model or switch to
Anthropic.

## Reproduce headline results (under 15 minutes)

```bash
cd src
python run_eval.py --limit 20     # quick smoke test, ~30 seconds
python run_eval.py                # full 201-example golden set, ~2-5 minutes
```

This prints intent-classification accuracy and routing (auto-handle vs
escalate) accuracy against the hand-labeled golden set, and saves
per-example predictions to `data/eval_predictions.csv`.

Note: `run_eval.py` reports basic accuracy only. The full evaluation harness
— baselines, LLM-as-judge for reply quality, and a human-agreement check for
the judge — is a separate deliverable (`eval_harness.py`, see report).

## Full evaluation harness (baselines + LLM judge + human agreement)

Two baselines run **fully offline, no key needed**, and already have real results:

```bash
cd src
python baselines.py
```

| Baseline | Intent accuracy | Routing accuracy |
|---|---|---|
| Trivial (always guess majority class) | 55.7% | 78.1% |
| Simple (keyword regex + nearest-neighbor reply) | 63.2% | 26.9% |

Note the simple baseline's routing accuracy is *worse* than the trivial one
-- its crude keyword rule wrongly calls most ambiguous cases AUTO_HANDLE,
while just guessing the majority class (ESCALATE) happens to be safer. Don't
let a "smarter-looking" baseline number fool you; see the report for more of
this kind of thing.

Then, with `ANTHROPIC_API_KEY` set, run the reply-quality judge and the
judge-vs-human agreement check:

```bash
python run_eval.py                                            # agent's own predictions
python judge.py --predictions ../data/eval_predictions.csv \
    --reply_col pred_reply --out ../data/judge_scores_agent.csv
python judge.py --predictions ../data/baseline_simple_predictions.csv \
    --reply_col pred_reply --out ../data/judge_scores_simple_baseline.csv
python judge.py --predictions ../data/human_score_candidates.csv \
    --reply_col pred_reply --out ../data/llm_judge_scores_on_human_subset.csv
python agreement.py
python eval_harness.py     # prints the full combined report
```

`data/human_judge_scores.csv` contains 30 replies hand-scored by the
assignment author against the same rubric the LLM judge uses (groundedness,
helpfulness, tone_fit, safety_pass) -- `agreement.py` compares the judge's
scores on those same 30 examples against these human scores (mean absolute
error, exact-match rate, Pearson correlation, weighted Cohen's kappa) so the
judge's reliability is evidenced, not assumed.

## Regenerating the corpus from scratch (optional, ~2 minutes)

You only need this if you want to rebuild `apple_opening_pairs_en.csv` from
the raw `twcs.csv` yourself (e.g. to pick a different brand). Download
`twcs.csv` from
[Kaggle: Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter),
then:

```bash
cd src
python build_corpus.py --raw ../data/twcs.csv --out ../data/apple_opening_pairs_en.csv
```

## Design notes worth knowing before reading the code

- **Retrieval, not fine-tuning.** Replies are grounded by retrieving the 3
  most similar historical AppleSupport resolutions (TF-IDF cosine similarity)
  and instructing the model not to invent URLs/instructions absent from those
  examples. This is deliberately simple and auditable over a heavier RAG setup.
- **Golden-set leakage is actively prevented.** The retrieval corpus excludes
  every golden-set example (by exact text match) so a prediction can never be
  grounded on retrieving its own ground-truth answer.
- **Routing is a stated policy, not a vibe.** See `ROUTING_POLICY` in
  `intents.py` — every routing decision must cite which policy clause applied.
- **The simple baseline is deliberately not upgraded.** Its keyword classifier
  is literally the first-draft heuristic from the labeling phase, kept
  as-is (not cleaned up) because a fair "simple baseline" should reflect what
  a naive first pass actually looks like, warts included.
- **A safety failure zeroes out the whole reply score.** In `judge.py`,
  `composite = mean(groundedness, helpfulness, tone_fit)` only if
  `safety_pass` is true, otherwise 0 — a beautifully-worded reply that leaks
  a security risk is not a good reply.
