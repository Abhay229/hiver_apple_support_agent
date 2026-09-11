"""
Retrieval over historical resolved AppleSupport conversations.

We use TF-IDF + cosine similarity rather than a neural embedding model on
purpose: it's fast, deterministic, needs no extra API calls/cost, and is
transparent enough to debug when a retrieved example looks wrong -- good
tradeoffs for a take-home-scale system. See decision log.

IMPORTANT: golden eval examples are excluded from the retrieval corpus so a
reply can never be grounded on retrieving its own ground-truth answer. This
is done by exact customer_text match, since the golden set doesn't carry the
original tweet_id through (a known limitation, logged in decision log).
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HistoricalRetriever:
    def __init__(self, corpus_csv: str, golden_csv: str | None = None, max_corpus_size: int | None = 30000):
        df = pd.read_csv(corpus_csv)

        if golden_csv is not None:
            golden_texts = set(pd.read_csv(golden_csv)["customer_text"].tolist())
            before = len(df)
            df = df[~df["customer_text"].isin(golden_texts)].reset_index(drop=True)
            print(f"[retrieval] excluded {before - len(df)} golden-set examples from corpus "
                  f"to avoid grounding a reply on its own answer")

        if max_corpus_size is not None and len(df) > max_corpus_size:
            df = df.sample(max_corpus_size, random_state=42).reset_index(drop=True)

        self.df = df
        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(df["customer_text"].astype(str))
        print(f"[retrieval] index built over {len(df)} historical resolved conversations")

    def retrieve(self, query: str, k: int = 3):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        results = []
        for i in top_idx:
            results.append({
                "customer_text": self.df.iloc[i]["customer_text"],
                "reply_text": self.df.iloc[i]["reply_text"],
                "similarity": float(sims[i]),
            })
        return results
