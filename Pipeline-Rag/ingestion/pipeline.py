"""
Pipeline de ingesta automatizado (Unidad 3):

    Documentos crudos → Extracción → Limpieza/Normalización → Transformación
    (chunking) → Embeddings → Almacenamiento vectorial

Incluye registro (logging) en cada etapa, tal como recomienda el material
del curso, para poder detectar y corregir fallas sin reprocesar todo el
pipeline.
"""
import logging
from pathlib import Path

from config.settings import settings
from ingestion.cleaners import clean_corpus
from ingestion.extractors import extract_directory
from rag.chunking import chunk_documents
from rag.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("ingestion-pipeline")


def run_pipeline(source_dir: str | None = None, reset: bool = False, space: str | None = None) -> dict:
    """Ejecuta el pipeline de ingesta completo y devuelve un resumen numérico."""
    source_path = Path(source_dir or settings.RAW_DATA_DIR)

    logger.info("Etapa 1/4 — Extracción: leyendo documentos desde '%s'", source_path)
    raw_docs = extract_directory(source_path)
    n_archivos = len({d.source for d in raw_docs})
    logger.info("  -> %d fragmento(s) crudo(s) extraído(s) de %d archivo(s)", len(raw_docs), n_archivos)

    logger.info("Etapa 2/4 — Limpieza y normalización")
    clean_docs = clean_corpus(raw_docs)
    logger.info("  -> %d documento(s) limpio(s) (encoding, marcado residual, duplicados)", len(clean_docs))

    logger.info("Etapa 3/4 — Transformación (chunking)")
    chunks = chunk_documents(
        clean_docs, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    )
    logger.info(
        "  -> %d chunk(s) generado(s) (tamaño=%d car., solapamiento=%d car.)",
        len(chunks), settings.CHUNK_SIZE, settings.CHUNK_OVERLAP,
    )

    logger.info("Etapa 4/4 — Embeddings + almacenamiento vectorial")
    store = VectorStore(space=space)
    if reset:
        logger.info("  -> Reiniciando la colección '%s' antes de insertar", store.collection.name)
        store.reset()
    # Cada fuente se reemplaza de forma idempotente: re-ingerir no duplica chunks, las
    # ediciones eliminan los chunks obsoletos y lo que no cambió no recalcula embeddings.
    por_fuente: dict[str, list] = {}
    for chunk in chunks:
        por_fuente.setdefault(chunk.source, []).append(chunk)
    reemplazos = [store.replace_source(fuente, lista) for fuente, lista in por_fuente.items()]
    sin_cambios = sum(1 for r in reemplazos if r["unchanged"])
    logger.info(
        "  -> %d fuente(s) sin cambios, %d actualizada(s)", sin_cambios, len(reemplazos) - sin_cambios
    )
    logger.info("  -> La colección '%s' contiene ahora %d chunk(s)", store.collection.name, store.count())

    return {
        "archivos_procesados": n_archivos,
        "fragmentos_extraidos": len(raw_docs),
        "documentos_limpios": len(clean_docs),
        "chunks_generados": len(chunks),
        "fuentes_sin_cambios": sin_cambios,
        "total_en_coleccion": store.count(),
    }


if __name__ == "__main__":
    import json

    resumen = run_pipeline(reset=True)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
