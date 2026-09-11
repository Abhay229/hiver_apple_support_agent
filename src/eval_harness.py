"""
Full evaluation harness: baselines + agent + LLM-judge reply quality + a
human-agreement check for the judge.

Run this after run_eval.py has produced data/eval_predictions.csv (the
agent's own predictions on the golden set).

Usage (all require ANTHROPIC_API_KEY except the baselines, which are offline):
    python baselines.py                                          # offline, no key needed
    python run_eval.py                                           # agent on golden set
    python judge.py --predictions ../data/eval_predictions.csv \\
        --reply_col pred_reply --out ../data/judge_scores_agent.csv
    python judge.py --predictions ../data/baseline_simple_predictions.csv \\
        --reply_col pred_reply --out ../data/judge_scores_simple_baseline.csv
    python judge.py --predictions ../data/human_score_candidates.csv \\
        --reply_col pred_reply --out ../data/llm_judge_scores_on_human_subset.csv
    python agreement.py                                          # judge vs human agreement
    python eval_harness.py                                       # this file: prints the full report
"""

import pandas as pd
import os


def safe_read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def main():
    print("=" * 60)
    print("HIVER TAKE-HOME: FULL EVALUATION REPORT")
    print("=" * 60)

    baseline_summary = safe_read("../data/baseline_summary.csv")
    if baseline_summary is not None:
        print("\n--- Baselines (intent / routing accuracy) ---")
        print(baseline_summary.to_string(index=False))
    else:
        print("\n[missing] run baselines.py first")

    judge_agent = safe_read("../data/judge_scores_agent.csv")
    judge_simple = safe_read("../data/judge_scores_simple_baseline.csv")
    if judge_agent is not None:
        print(f"\n--- Agent reply quality (LLM judge, n={len(judge_agent)}) ---")
        print(f"Mean composite: {judge_agent['composite'].mean():.2f} / 5")
        print(f"Safety pass rate: {judge_agent['safety_pass'].mean():.1%}")
    if judge_simple is not None:
        print(f"\n--- Simple baseline reply quality (LLM judge, n={len(judge_simple)}) ---")
        print(f"Mean composite: {judge_simple['composite'].mean():.2f} / 5")
        print(f"Safety pass rate: {judge_simple['safety_pass'].mean():.1%}")

    agreement = safe_read("../data/judge_agreement_report.csv")
    if agreement is not None:
        print("\n--- Judge vs. human agreement (n=30 hand-labeled) ---")
        print(agreement.to_string(index=False))
    else:
        print("\n[missing] run judge.py on human_score_candidates.csv, then agreement.py")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
