"""
Rebuilds data/apple_opening_pairs_en.csv from the raw Kaggle twcs.csv.

This is provided for reproducibility, but the output is already committed to
the repo (~22MB) so you don't need to run this or re-download the 516MB raw
file just to reproduce the headline results -- see README.

Usage:
    python build_corpus.py --raw ../data/twcs.csv --out ../data/apple_opening_pairs_en.csv
"""

import argparse
import pandas as pd


def is_probably_english(text: str) -> bool:
    t = str(text)
    non_ascii = sum(ord(c) > 127 for c in t)
    return non_ascii / max(len(t), 1) < 0.15


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="../data/twcs.csv")
    parser.add_argument("--out", default="../data/apple_opening_pairs_en.csv")
    parser.add_argument("--brand", default="AppleSupport")
    args = parser.parse_args()

    df = pd.read_csv(args.raw, dtype={"tweet_id": "int64"}, low_memory=False)
    df["in_response_to_tweet_id"] = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce")
    by_id = df.set_index("tweet_id", drop=False)

    replies = df[(df["inbound"] == False) & (df["author_id"] == args.brand)].copy()
    replies = replies.dropna(subset=["in_response_to_tweet_id"])
    replies["in_response_to_tweet_id"] = replies["in_response_to_tweet_id"].astype("int64")

    opening_pairs = []
    for reply_idx in replies.index:
        cust_id = replies.loc[reply_idx, "in_response_to_tweet_id"]
        if cust_id in by_id.index:
            cust_row = by_id.loc[cust_id]
            if isinstance(cust_row, pd.DataFrame):
                cust_row = cust_row.iloc[0]
            # thread-starter = inbound tweet that is not itself a reply
            if cust_row["inbound"] == True and pd.isna(cust_row["in_response_to_tweet_id"]):
                opening_pairs.append({
                    "customer_tweet_id": int(cust_id),
                    "customer_text": cust_row["text"],
                    "reply_text": replies.loc[reply_idx, "text"],
                    "created_at": cust_row["created_at"],
                })

    out_df = pd.DataFrame(opening_pairs).drop_duplicates(subset="customer_tweet_id")
    out_df = out_df[out_df["customer_text"].apply(is_probably_english)]
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} opening-message -> first-reply pairs to {args.out}")


if __name__ == "__main__":
    main()
