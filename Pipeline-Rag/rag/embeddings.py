"""
Generación de embeddings 100% local con `sentence-transformers`.

El modelo se descarga una única vez desde Hugging Face Hub (open source,
gratuito) y luego corre en CPU/GPU local sin ninguna llamada de red
adicional ni costo por uso.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config.settings import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Carga (una sola vez, con caché) el modelo de embeddings configurado."""
    try:
        return SentenceTransformer(settings.EMBEDDING_MODEL)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"No se pudo cargar el modelo de embeddings '{settings.EMBEDDING_MODEL}'.\n"
            "La primera vez que corres el proyecto, este modelo se descarga "
            "automáticamente y de forma gratuita desde Hugging Face Hub (~470MB). "
            "Verifica tu conexión a internet e inténtalo de nuevo.\n"
            f"Detalle técnico: {exc}"
        ) from exc


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Genera embeddings normalizados (para similitud coseno) de una lista de textos."""
    if not texts:
        return []
    modelo = get_embedding_model()
    vectores = modelo.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectores.tolist()


def embed_query(text: str) -> list[float]:
    """Genera el embedding de una única consulta."""
    return embed_texts([text])[0]
