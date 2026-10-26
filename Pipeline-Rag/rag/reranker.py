"""
Reranking local con un cross-encoder (P2-1, experimental y opcional).

La recuperación híbrida trae `candidates` fragmentos y el cross-encoder, que lee la pregunta y
cada fragmento JUNTOS, los reordena; se devuelven los `top_k` mejores. Corre en CPU, sin red
una vez descargado el modelo. Los modelos se cargan una sola vez por proceso.

Modelos evaluados (licencias verificadas en sus fichas oficiales de Hugging Face, 2026-10-19):
  BAAI/bge-reranker-v2-m3                       Apache-2.0 · 0.6B parámetros · multilingüe
  cross-encoder/mmarco-mMiniLMv2-L12-H384-v1    Apache-2.0 · 0.1B parámetros · la ficha no nombra el español
"""
from functools import lru_cache

from rag.vector_store import RetrievedChunk


@lru_cache(maxsize=2)
def get_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    try:
        return CrossEncoder(model_name)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"No se pudo cargar el reranker '{model_name}'. La primera vez se descarga desde Hugging Face Hub; "
            f"verifica la conexión. Detalle técnico: {exc}"
        ) from exc


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int, model_name: str, predict=None) -> list[RetrievedChunk]:
    """Reordena `chunks` por relevancia según el cross-encoder y devuelve los `top_k` mejores.
    `predict` es inyectable ((consulta, texto) -> puntaje) para probar la lógica sin el modelo."""
    if not chunks:
        return []
    pares = [(query, c.text) for c in chunks]
    if predict is None:
        puntajes = [float(s) for s in get_reranker(model_name).predict(pares, show_progress_bar=False)]
    else:
        puntajes = [float(predict(q, t)) for q, t in pares]
    orden = sorted(range(len(chunks)), key=lambda i: (-puntajes[i], i))[:top_k]
    salida = []
    for i in orden:
        c = chunks[i]
        salida.append(RetrievedChunk(
            text=c.text, source=c.source, page=c.page, score=puntajes[i], chunk_id=c.chunk_id,
            doc_type=c.doc_type, method="hybrid_rerank",
            scores={**c.scores, "rerank": puntajes[i], "pre_rank": i + 1},
        ))
    return salida
