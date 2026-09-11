"""
Two baselines to compare the LLM agent against. Both run fully offline --
no API key needed -- so their numbers are available immediately.

TRIVIAL baseline: what a support-ops lead would do with zero analysis --
always guess the most obviously common intent, always escalate (since most
messages need a human anyway), always send one canned reply.

SIMPLE baseline: a naive but real approach a junior engineer might ship in a
day -- priority-ordered keyword regex for intent (this is literally our
first-draft classifier from the labeling phase, which manual review found
wrong ~40% of the time -- see labeling_note.md) + retrieve-and-copy the single
most similar historical reply verbatim (no generation) + a crude escalation
rule based on sensitive keywords / shouting, written independently from the
policy used to build the gold labels (to avoid a circular, leaking baseline).
"""

import re
import pandas as pd
from retrieval import HistoricalRetriever

# ---------------------------------------------------------------------------
# TRIVIAL BASELINE
# ---------------------------------------------------------------------------

TRIVIAL_INTENT = "software_bug_or_update"          # obviously the biggest bucket
TRIVIAL_ROUTING = "ESCALATE"                        # majority class in gold set (78%)
TRIVIAL_REPLY = ("Thanks for reaching out! We're here to help -- please send us a "
                  "DM with more details so we can look into this. https://t.co/GDrqU22YpT")


def run_trivial_baseline(golden_df: pd.DataFrame) -> pd.DataFrame:
    out = golden_df.copy()
    out["pred_intent"] = TRIVIAL_INTENT
    out["pred_routing"] = TRIVIAL_ROUTING
    out["pred_reply"] = TRIVIAL_REPLY
    return out


# ---------------------------------------------------------------------------
# SIMPLE BASELINE
# ---------------------------------------------------------------------------

# This is the ORIGINAL priority-ordered keyword classifier from the labeling
# phase, kept as-is (warts and all) rather than upgraded, because that's the
# honest point of a "simple" baseline. See labeling_note.md for its known
# ~40% error rate found during manual review.
_INTENT_RULES = [
    ("warranty_repair",   r"\b(warranty|repair|service cent(er|re)|genius bar|replace(ment|d)?\b.*(cover|store)|not covered)\b"),
    ("hardware_defect",   r"\b(crack(ed)?|shatter|screen|swoll?en|shock(ed)?|speaker|button (stuck|broken)|charging port|physically|scratch)\b"),
    ("account_subscription", r"\b(apple ?id|icloud|touch ?id|face ?id|subscription|apple music|log(ged)? ?in|logged out|password|account|purchase history|(photo|picture)s? (all )?(gone|missing|disappear|deleted))\b"),
    ("connectivity_compat", r"\b(bluetooth|car ?play|\bsim\b|carrier|imessage|wi-?fi|pair(ing)?|connect(ing|ed|ion)?)\b"),
    ("how_to_question",   r"\b(how do i|how to|is there a way|any way to|can i |how can i)\b"),
    ("software_bug_or_update", r"\b(updat|ios ?1\d|freez|glitch|crash|slow|lag|battery|bug|autocorrect|keyboard|capitali[sz]|question mark|weird|malfunction|broken (after|since)|stopped working|not working since|randomly|letter i\b)\b"),
]


def keyword_classify(text: str) -> str:
    tl = str(text).lower()
    for label, pat in _INTENT_RULES:
        if re.search(pat, tl):
            return label
    return "general_complaint_venting"


# Deliberately independent from ROUTING_POLICY in intents.py -- a crude rule
# an ops team might write without the benefit of reading 200 examples by hand.
_SENSITIVE_KEYWORDS = ["password", "apple id", "icloud", "charge", "charged", "refund",
                        "warranty", "repair", "serial", "broken screen", "genius bar"]


def keyword_route(text: str) -> str:
    tl = str(text).lower()
    if any(k in tl for k in _SENSITIVE_KEYWORDS):
        return "ESCALATE"
    shouting = sum(1 for c in text if c.isupper()) / max(len(text), 1) > 0.3
    if shouting:
        return "ESCALATE"
    return "AUTO_HANDLE"


def run_simple_baseline(golden_df: pd.DataFrame, retriever: HistoricalRetriever) -> pd.DataFrame:
    out = golden_df.copy()
    out["pred_intent"] = out["customer_text"].apply(keyword_classify)
    out["pred_routing"] = out["customer_text"].apply(keyword_route)
    out["pred_reply"] = out["customer_text"].apply(
        lambda t: retriever.retrieve(t, k=1)[0]["reply_text"]
    )
    return out


# ---------------------------------------------------------------------------

def score(name: str, df: pd.DataFrame):
    intent_acc = (df["final_intent"] == df["pred_intent"]).mean()
    routing_acc = (df["routing"] == df["pred_routing"]).mean()
    print(f"\n=== {name} ===")
    print(f"Intent accuracy:  {intent_acc:.1%}")
    print(f"Routing accuracy: {routing_acc:.1%}")
    return {"baseline": name, "intent_acc": intent_acc, "routing_acc": routing_acc}


if __name__ == "__main__":
    golden = pd.read_csv("../data/golden_eval_set.csv")
    retriever = HistoricalRetriever(
        corpus_csv="../data/apple_opening_pairs_en.csv",
        golden_csv="../data/golden_eval_set.csv",
    )

    trivial_out = run_trivial_baseline(golden)
    simple_out = run_simple_baseline(golden, retriever)

    trivial_out.to_csv("../data/baseline_trivial_predictions.csv", index=False)
    simple_out.to_csv("../data/baseline_simple_predictions.csv", index=False)

    results = [score("TRIVIAL baseline", trivial_out), score("SIMPLE baseline", simple_out)]
    pd.DataFrame(results).to_csv("../data/baseline_summary.csv", index=False)
