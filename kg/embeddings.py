from __future__ import annotations

import os
import threading
from array import array
from typing import Sequence


EMBEDDING_MODEL = os.environ.get(
    "KG_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
)

_lock = threading.Lock()
_model = None
_cache: dict[str, array[float]] = {}


def cosine_scores(query: str, passages: Sequence[str]) -> list[float]:
    """Return multilingual E5 cosine similarities for one query and passages."""
    if not passages:
        return []
    texts = [f"query: {query}", *(f"passage: {item}" for item in passages)]
    vectors = _encode(texts)
    query_vector = vectors[0]
    return [
        sum(left * right for left, right in zip(query_vector, vector))
        for vector in vectors[1:]
    ]


def _encode(texts: Sequence[str]) -> list[array[float]]:
    missing = list(dict.fromkeys(text for text in texts if text not in _cache))
    with _lock:
        missing = [text for text in missing if text not in _cache]
        if missing:
            model = _get_model()
            vectors = model.encode(
                missing,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for text, vector in zip(missing, vectors):
                # float32 keeps a full-book entity index compact in memory.
                _cache[text] = array("f", vector)
    return [_cache[text] for text in texts]


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - packaging guard
            raise RuntimeError(
                "实体候选召回需要 sentence-transformers；请重新安装项目依赖"
            ) from exc
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model
