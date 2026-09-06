"""Optional cross-encoder reranker (ONNX, torch-free).

Off by default — on a CPU-only laptop the latency isn't worth it. Enable with
``ENABLE_RERANK=true`` once a GPU is available, or to trade speed for accuracy.
Install extra: ``pip install fastembed``.
"""

from __future__ import annotations

import functools

_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


@functools.lru_cache(maxsize=1)
def _reranker():  # noqa: ANN202
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=_MODEL)


def cross_encoder_scores(query: str, passages: list[str]) -> list[float]:
    return list(_reranker().rerank(query, passages))
