"""
Runs the SupportAgent over the golden evaluation set and reports basic
intent-classification and routing accuracy against the hand-labeled ground
truth.

This is NOT the full evaluation harness (that also needs an LLM-as-judge for
reply quality + a human-agreement check, and comparison against baselines --
see eval_harness.py for that). This script is the fast sanity check: does the
classifier/router even work.

Usage:
    export GROQ_API_KEY=gsk_...        # or ANTHROPIC_API_KEY with LLM_PROVIDER=anthropic
    python run_eval.py --limit 20      # quick check on 20 examples (~30s)
    python run_eval.py                 # full 201-example run

Resumable: results are checkpointed to --out after every example, so if the
run is interrupted (rate limit exhausted, Ctrl-C, crash) you can re-run the
exact same command and it will skip everything already completed and only
fetch what's missing. A row that fails after all retries is recorded as
ROW_ERROR (not silently dropped, not a crash) so one bad example can't cost
you the rest of the run.
"""

import argparse
import concurrent.futures as cf
import pandas as pd
from tqdm import tqdm

from agent import SupportAgent
from retrieval import HistoricalRetriever

DATA_DIR = "../data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run on first N golden examples")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent API calls (keep low on free tiers)")
    parser.add_argument("--out", type=str, default=f"{DATA_DIR}/eval_predictions.csv")
    parser.add_argument("--retry-errors", action="store_true",
                         help="Also re-run rows already checkpointed as ROW_ERROR")
    args = parser.parse_args()

    golden = pd.read_csv(f"{DATA_DIR}/golden_eval_set.csv")
    if args.limit:
        golden = golden.head(args.limit)

    # --- resume support: load whatever's already checkpointed ---
    try:
        done_df = pd.read_csv(args.out)
        done_ids = set(done_df["gold_id"])
        if args.retry_errors:
            error_ids = set(done_df.loc[done_df["pred_intent"] == "ROW_ERROR", "gold_id"])
            done_ids -= error_ids
            done_df = done_df[~done_df["gold_id"].isin(error_ids)]
        results = done_df.to_dict("records")
        print(f"Resuming: {len(done_ids)} examples already done in {args.out}, skipping those.")
    except FileNotFoundError:
        done_ids, results = set(), []

    rows = [r for r in golden.to_dict("records") if r["gold_id"] not in done_ids]
    if not rows:
        print("Nothing left to run — all examples already checkpointed.")
    else:
        retriever = HistoricalRetriever(
            corpus_csv=f"{DATA_DIR}/apple_opening_pairs_en.csv",
            golden_csv=f"{DATA_DIR}/golden_eval_set.csv",
        )
        agent = SupportAgent(retriever)

        def run_one(row):
            try:
                result = agent.handle(row["customer_text"])
            except Exception as exc:
                # llm_client already retried transient errors internally;
                # if we're still here, record it and move on rather than
                # taking down the whole run.
                result = {
                    "intent": "ROW_ERROR", "intent_confidence": "low",
                    "routing": "ESCALATE",
                    "routing_reason": f"Agent call failed after retries: {type(exc).__name__}: {exc}",
                    "draft_reply": "",
                }
            return {
                "gold_id": row["gold_id"],
                "customer_text": row["customer_text"],
                "gold_intent": row["final_intent"],
                "pred_intent": result.get("intent"),
                "gold_routing": row["routing"],
                "pred_routing": result.get("routing"),
                "pred_routing_reason": result.get("routing_reason"),
                "pred_reply": result.get("draft_reply"),
                "pred_confidence": result.get("intent_confidence"),
            }

        with cf.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_one, row): row for row in rows}
            for f in tqdm(cf.as_completed(futures), total=len(futures), desc="Running agent"):
                results.append(f.result())
                # checkpoint after every completion so interruptions cost nothing
                pd.DataFrame(results).to_csv(args.out, index=False)

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, index=False)

    n_errors = (out_df["pred_intent"] == "ROW_ERROR").sum()
    if n_errors:
        print(f"\n{n_errors} row(s) failed after retries (ROW_ERROR) — "
              f"re-run with --retry-errors to retry just those.")

    intent_acc = (out_df["gold_intent"] == out_df["pred_intent"]).mean()
    routing_acc = (out_df["gold_routing"] == out_df["pred_routing"]).mean()

    print(f"\nRan on {len(out_df)} examples")
    print(f"Intent accuracy:  {intent_acc:.1%}")
    print(f"Routing accuracy: {routing_acc:.1%}")
    print(f"Predictions saved to {args.out}")
    print("\nPer-intent breakdown:")
    out_df["_correct"] = out_df["gold_intent"] == out_df["pred_intent"]
    print(out_df.groupby("gold_intent")["_correct"].mean().sort_values())


if __name__ == "__main__":
    main()