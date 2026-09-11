"""
Computes how well the LLM judge agrees with a human on the same 30 examples.

This is the evidence required by the assignment that the judge isn't just
hallucinating plausible-looking scores. Run judge.py on
data/human_score_candidates.csv first to produce the LLM's scores, then run
this to compare against data/human_judge_scores.csv (hand-labeled by the
assignment author, see labeling_note.md-style reasoning inline in that file).

Usage:
    python judge.py --predictions ../data/human_score_candidates.csv \\
        --reply_col pred_reply --out ../data/llm_judge_scores_on_human_subset.csv
    python agreement.py
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

DIMENSIONS = ["groundedness", "helpfulness", "tone_fit"]


def cohen_kappa_ordinal(a: np.ndarray, b: np.ndarray, max_score: int = 5) -> float:
    """Linearly-weighted Cohen's kappa for ordinal 1-5 scores."""
    n = len(a)
    scores = list(range(1, max_score + 1))
    confusion = np.zeros((max_score, max_score))
    for x, y in zip(a, b):
        confusion[int(x) - 1, int(y) - 1] += 1
    row_marg = confusion.sum(axis=1)
    col_marg = confusion.sum(axis=0)
    expected = np.outer(row_marg, col_marg) / n
    weights = np.array([[1 - abs(i - j) / (max_score - 1) for j in range(max_score)] for i in range(max_score)])
    observed_agreement = (confusion * weights).sum() / n
    expected_agreement = (expected * weights).sum() / n
    if expected_agreement == 1:
        return 1.0
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def main(human_csv="../data/human_judge_scores.csv",
         llm_csv="../data/llm_judge_scores_on_human_subset.csv"):
    human = pd.read_csv(human_csv)
    llm = pd.read_csv(llm_csv)
    merged = human.merge(llm, on="gold_id", suffixes=("_human", "_llm"))

    print(f"Comparing on {len(merged)} shared examples\n")
    report = []
    for dim in DIMENSIONS:
        h = merged[f"human_{dim}"].astype(float).values
        l = merged[dim].astype(float).values
        mae = np.mean(np.abs(h - l))
        exact_match = np.mean(h == l)
        within_1 = np.mean(np.abs(h - l) <= 1)
        corr, _ = pearsonr(h, l)
        kappa = cohen_kappa_ordinal(h, l)
        report.append({
            "dimension": dim, "mean_abs_error": round(mae, 2),
            "exact_match_rate": round(exact_match, 2),
            "within_1_point_rate": round(within_1, 2),
            "pearson_r": round(corr, 2),
            "weighted_kappa": round(kappa, 2),
        })
        print(f"{dim:15s} MAE={mae:.2f}  exact={exact_match:.0%}  within_1={within_1:.0%}  "
              f"r={corr:.2f}  weighted_kappa={kappa:.2f}")

    # Safety gate agreement (binary)
    safety_agree = (merged["human_safety_pass"].astype(str).str.lower()
                     == merged["safety_pass"].astype(str).str.lower()).mean()
    print(f"\nSafety-gate agreement: {safety_agree:.0%}")

    out = pd.DataFrame(report)
    out.to_csv("../data/judge_agreement_report.csv", index=False)
    print("\nSaved -> ../data/judge_agreement_report.csv")
    return out


if __name__ == "__main__":
    main()
