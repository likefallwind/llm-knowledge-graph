from __future__ import annotations

from collections import Counter
from unittest import mock

import pytest


def _fake_embedding_scores(query: str, passages: list[str]) -> list[float]:
    """Deterministic test stand-in that avoids downloading the real model."""

    def vector(text: str) -> Counter[str]:
        return Counter(char.casefold() for char in text if char.isalnum())

    query_vector = vector(query)
    query_norm = sum(value * value for value in query_vector.values()) ** 0.5
    scores = []
    for passage in passages:
        passage_vector = vector(passage)
        passage_norm = sum(
            value * value for value in passage_vector.values()
        ) ** 0.5
        dot = sum(
            value * passage_vector.get(char, 0)
            for char, value in query_vector.items()
        )
        scores.append(dot / (query_norm * passage_norm) if passage_norm else 0.0)
    return scores


@pytest.fixture(autouse=True)
def no_embedding_model_download():
    with mock.patch(
        "kg.embeddings.cosine_scores", side_effect=_fake_embedding_scores
    ):
        yield
