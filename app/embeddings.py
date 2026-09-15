"""Pluggable text-embedding interface.

`Embedder` is the contract the rest of the pipeline relies on. The default
implementation, `TfidfEmbedder`, needs no external API calls or model
downloads, which keeps ingestion/retrieval fast, deterministic, and testable
offline. Swap in a dense embedder (sentence-transformers, OpenAI, Voyage,
etc.) by implementing the same interface and updating
`get_embedder()`/`EMBEDDER_BACKEND`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Embedder(ABC):
    """Contract: fit on a collection's chunks, then embed queries against it."""

    @abstractmethod
    def fit(self, texts: List[str]) -> np.ndarray:
        """Fit the embedder on `texts` and return their embedding matrix."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string using the already-fit vocabulary/model."""

    @abstractmethod
    def to_bytes(self) -> bytes:
        """Serialize fitted state so it can be persisted alongside a collection."""

    @classmethod
    @abstractmethod
    def from_bytes(cls, data: bytes) -> "Embedder":
        """Rehydrate a fitted embedder from `to_bytes()` output."""


class TfidfEmbedder(Embedder):
    """TF-IDF + cosine-similarity embedder backed by scikit-learn.

    This is a lexical (bag-of-words) representation, not a semantic/dense
    one. It's chosen as the default because it requires no network access,
    no GPU, and no model download — it works the moment scikit-learn is
    installed, which makes ingestion and retrieval trivially unit-testable.
    """

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50_000,
        )
        self._fitted = False

    def fit(self, texts: List[str]) -> np.ndarray:
        matrix = self._vectorizer.fit_transform(texts)
        self._fitted = True
        return matrix.toarray().astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Embedder must be fit before embedding queries.")
        vec = self._vectorizer.transform([text])
        return vec.toarray().astype(np.float32)[0]

    def to_bytes(self) -> bytes:
        import pickle

        return pickle.dumps(self._vectorizer)

    @classmethod
    def from_bytes(cls, data: bytes) -> "TfidfEmbedder":
        import pickle

        instance = cls.__new__(cls)
        instance._vectorizer = pickle.loads(data)
        instance._fitted = True
        return instance


def get_embedder(backend: str = "tfidf") -> Embedder:
    if backend == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"Unknown embedder backend: {backend}")
