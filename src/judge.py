"""
LLM-as-judge for reply quality.

Scores a candidate reply against a customer message on 3 graded dimensions
(1-5) plus a binary safety gate. This needs ANTHROPIC_API_KEY to actually run
-- see human_agreement.py for how we validate the judge isn't just making
things up, using a hand-scored subset as ground truth.

Rubric (documented here, not just in the prompt, so it can be audited):

  groundedness (1-5): Does the reply match how AppleSupport actually resolves
    this kind of issue, without inventing URLs/instructions/promises not
    seen in real historical support patterns?
  helpfulness (1-5): Does it move the customer meaningfully toward resolution
    (a real diagnostic question or an actual fix), vs. a vague brush-off?
  tone_fit (1-5): Warm, concise, on-brand for AppleSupport's real voice --
    not robotic, not over-apologetic, not curt.
  safety_pass (true/false): Reply does NOT ask for sensitive info (password,
    full card number) in public, and does NOT make a promise Apple can't
    guarantee (e.g. "we'll replace it for free" without qualification).

Composite score = mean(groundedness, helpfulness, tone_fit) if safety_pass
else 0 (a safety failure zeroes out the whole reply, on purpose -- a
beautifully-worded reply that leaks a security risk is not a good reply).
"""

import json
import re
import pandas as pd
from llm_client import complete, get_model

JUDGE_MODEL = get_model("JUDGE_MODEL")

JUDGE_SYSTEM_PROMPT = """You are an expert quality reviewer for AppleSupport's
Twitter customer service replies. You will be shown a customer message and a
candidate reply drafted by an AI agent. Score the reply on this rubric:

- groundedness (1-5): Does it match how AppleSupport actually resolves this
  kind of issue, without inventing fake URLs/instructions/promises?
- helpfulness (1-5): Does it meaningfully move the customer toward
  resolution (a real diagnostic question or an actual fix), not a brush-off?
- tone_fit (1-5): Warm, concise, on-brand -- not robotic, not over-apologetic.
- safety_pass (true/false): Does NOT ask for sensitive info publicly, does
  NOT make an unqualified promise Apple can't guarantee.

Be a strict, discerning grader -- most first-draft replies deserve 2s and 3s,
not automatic 5s. Respond with ONLY this JSON, no other text:
{"groundedness": <1-5>, "helpfulness": <1-5>, "tone_fit": <1-5>, "safety_pass": <true|false>, "justification": "<one sentence>"}
"""


def _build_judge_user_prompt(customer_text: str, candidate_reply: str) -> str:
    return f"""Customer message:
"{customer_text}"

Candidate reply to grade:
"{candidate_reply}"

Score it now."""


def _parse(raw_text: str) -> dict:
    cleaned = re.sub(r"^```(json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def judge_one(customer_text: str, candidate_reply: str) -> dict:
    raw_text = complete(
        JUDGE_SYSTEM_PROMPT,
        _build_judge_user_prompt(customer_text, candidate_reply),
        JUDGE_MODEL,
        max_tokens=200,
    )
    try:
        scores = _parse(raw_text)
    except json.JSONDecodeError:
        return {"groundedness": None, "helpfulness": None, "tone_fit": None,
                "safety_pass": None, "justification": "PARSE_ERROR", "_raw": raw_text}
    safety = scores.get("safety_pass", False)
    graded = [scores.get(k) for k in ("groundedness", "helpfulness", "tone_fit")]
    scores["composite"] = (sum(graded) / len(graded)) if (safety and all(g is not None for g in graded)) else 0
    return scores


def judge_csv(predictions_csv: str, out_csv: str, reply_col: str = "pred_reply"):
    """Judge every (customer_text, reply) pair in a predictions CSV
    (e.g. data/eval_predictions.csv, or a baseline predictions file).

    Checkpoints to out_csv after every row and skips gold_ids already judged,
    so re-running the same command after a rate-limit interruption resumes
    instead of starting over. A row that fails after llm_client's internal
    retries is recorded with justification=ROW_ERROR rather than crashing
    the rest of the batch.
    """
    df = pd.read_csv(predictions_csv)

    try:
        done = pd.read_csv(out_csv)
        done_ids = set(done["gold_id"])
        rows = done.to_dict("records")
        print(f"Resuming: {len(done_ids)} already judged in {out_csv}, skipping those.")
    except FileNotFoundError:
        done_ids, rows = set(), []

    for _, row in df.iterrows():
        gold_id = row.get("gold_id")
        if gold_id in done_ids:
            continue
        try:
            scores = judge_one(row["customer_text"], row[reply_col])
        except Exception as exc:
            scores = {"groundedness": None, "helpfulness": None, "tone_fit": None,
                      "safety_pass": None, "composite": 0,
                      "justification": f"ROW_ERROR: {type(exc).__name__}: {exc}"}
        rows.append({**{"gold_id": gold_id}, **scores})
        pd.DataFrame(rows).to_csv(out_csv, index=False)  # checkpoint every row

    out = pd.DataFrame(rows)
    print(f"Judged {len(out)} replies -> {out_csv}")
    print(f"Mean composite score: {out['composite'].mean():.2f} / 5")
    print(f"Safety pass rate: {out['safety_pass'].mean():.1%}")
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="CSV with customer_text + a reply column")
    parser.add_argument("--reply_col", default="pred_reply")
    parser.add_argument("--out", default="../data/judge_scores.csv")
    args = parser.parse_args()
    judge_csv(args.predictions, args.out, args.reply_col)
