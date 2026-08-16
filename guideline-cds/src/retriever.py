"""TF-IDF retrieval over the built-in corpus — fully offline, no downloads.

We use scikit-learn's TF-IDF vectorizer plus cosine similarity rather than a
neural embedding model, so the smoke test needs no model weights and no network.
For a real deployment over the full `epfl-llm/guidelines` corpus you would swap this
for a stronger retriever (for example a biomedical embedding model with a lexical
BM25 signal and a cross-encoder reranker); the interface below stays the same.
"""
from __future__ import annotations

from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .corpus import corpus_as_dicts


class Retriever:
    def __init__(self, documents: Optional[list[dict]] = None,
                 min_score: float = 0.05):
        """documents: list of {id,title,authority,license,url,text}.
        min_score: cosine floor below which a hit is treated as no match, so an
        off-topic question retrieves nothing and the flow abstains."""
        self.docs = documents if documents is not None else corpus_as_dicts()
        self.min_score = min_score
        # Fit over title + text so titles help match short queries.
        self._corpus_text = [f"{d['title']}. {d['text']}" for d in self.docs]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(self._corpus_text)

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Return up to k documents scoring above min_score, most relevant first.
        Each returned dict is a copy of the source doc plus a 'score' field."""
        if not query or not query.strip():
            return []
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        hits = []
        for i in ranked[:k]:
            score = float(sims[i])
            if score < self.min_score:
                continue
            hit = dict(self.docs[i])
            hit["score"] = score
            hits.append(hit)
        return hits
