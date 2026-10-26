"""Capa de servicios de la UI: conecta la interfaz con el backend real. Sin presentación."""
import tempfile
from pathlib import Path

from config.settings import settings
from ingestion.cleaners import clean_corpus
from ingestion.extractors import EXTRACTORS, extract_document
from rag.chunking import chunk_documents
from rag.spaces import DEFAULT_SPACE, list_spaces, validate_space_name
from rag.vector_store import VectorStore

SUPPORTED_EXTENSIONS = tuple(sorted(EXTRACTORS))


class UploadError(ValueError):
    """Error de carga con un mensaje accionable para el usuario."""


def open_store(space: str | None) -> VectorStore:
    return VectorStore(space=None if space in (None, DEFAULT_SPACE) else space)


def spaces(store: VectorStore) -> list[str]:
    nombres = [e["space"] for e in list_spaces(store.client)]
    return nombres if DEFAULT_SPACE in nombres else [DEFAULT_SPACE] + nombres


def create_space(name: str) -> str:
    nombre = validate_space_name(name)
    open_store(nombre)  # get_or_create
    return nombre


def ingest_upload(store: VectorStore, filename: str, data: bytes) -> dict:
    """Extrae, limpia, trocea e indexa un archivo subido. Devuelve {source, n_chunks, ...}."""
    nombre = Path(filename).name
    ext = Path(nombre).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UploadError(
            f"«{nombre}» no se puede cargar: el formato {ext or '(sin extensión)'} no está soportado. "
            f"Use un archivo {', '.join(SUPPORTED_EXTENSIONS)}."
        )
    if not data:
        raise UploadError(f"«{nombre}» está vacío.")
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / nombre
        ruta.write_bytes(data)
        try:
            crudos = extract_document(ruta)
        except Exception as exc:  # noqa: BLE001  (PDF corrupto, HTML ilegible, etc.)
            raise UploadError(f"No se pudo leer «{nombre}»: {str(exc)[:160]}") from exc
    limpios = clean_corpus(crudos)
    chunks = chunk_documents(limpios, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    if not chunks:
        raise UploadError(f"«{nombre}» no contiene texto extraíble (¿PDF escaneado? el OCR no está soportado).")
    resultado = store.replace_source(nombre, chunks)
    return {"source": nombre, "n_chunks": len(chunks), **resultado}


def delete_source(store: VectorStore, source: str) -> int:
    return store.delete_source(source)
