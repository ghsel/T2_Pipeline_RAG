"""
Chunking: divide el texto ya limpio en fragmentos ("chunks") de tamaño
manejable para generar embeddings y para caber en el contexto del LLM.

Se implementa un splitter recursivo: intenta cortar primero por los
separadores "más naturales" (párrafo → línea → oración → palabra), y solo
recurre a un corte "duro" por longitud si ningún separador funciona. Esto
evita partir una oración a la mitad siempre que sea posible.

Cada chunk conserva la metadata de su documento de origen (archivo y
página), que es lo que luego permite citar la fuente exacta en la
respuesta generada (Unidad 1: "Citas verificables").
"""
import hashlib
from dataclasses import dataclass

from ingestion.extractors import RawDocument

# Separadores probados en orden, del más "natural" al menos natural.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int | None
    doc_type: str


def make_chunk_id(source: str, page: int | None, position: int, text: str) -> str:
    """ID determinista: la misma fuente, página, posición y texto producen siempre
    el mismo ID. Así re-ingerir un documento no duplica sus chunks (upsert)."""
    huella = hashlib.sha1(f"{source}|{page}|{position}|{text}".encode("utf-8")).hexdigest()
    return huella[:16]


def _split_recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Divide `text` recursivamente probando cada separador de la lista."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # Sin separadores disponibles: corte duro por tamaño.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, *resto_separadores = separators
    partes = text.split(sep)

    fragmentos: list[str] = []
    actual = ""
    for parte in partes:
        candidato = (actual + sep + parte) if actual else parte
        if len(candidato) <= chunk_size:
            actual = candidato
        else:
            if actual:
                fragmentos.append(actual)
            if len(parte) > chunk_size:
                # La parte individual sigue siendo muy grande: recursión con
                # el siguiente separador de la lista.
                fragmentos.extend(_split_recursive(parte, chunk_size, resto_separadores))
                actual = ""
            else:
                actual = parte
    if actual:
        fragmentos.append(actual)

    return fragmentos


def _add_overlap(fragmentos: list[str], chunk_overlap: int) -> list[str]:
    """Añade solapamiento entre chunks consecutivos, tomando los últimos
    `chunk_overlap` caracteres del chunk anterior como prefijo del siguiente."""
    if chunk_overlap <= 0 or len(fragmentos) <= 1:
        return fragmentos

    con_overlap = [fragmentos[0]]
    for anterior, actual in zip(fragmentos, fragmentos[1:]):
        prefijo = anterior[-chunk_overlap:]
        con_overlap.append(f"{prefijo} {actual}".strip())
    return con_overlap


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    """Punto de entrada: divide un texto en fragmentos de ~chunk_size
    caracteres, con `chunk_overlap` caracteres de solapamiento entre ellos."""
    fragmentos = _split_recursive(text, chunk_size, DEFAULT_SEPARATORS)
    fragmentos = [f.strip() for f in fragmentos if f.strip()]
    return _add_overlap(fragmentos, chunk_overlap)


def chunk_documents(
    documents: list[RawDocument], chunk_size: int = 800, chunk_overlap: int = 120
) -> list[Chunk]:
    """Aplica `split_text` a una colección completa de documentos limpios,
    devolviendo chunks con su metadata de trazabilidad."""
    chunks: list[Chunk] = []
    posiciones: dict[tuple[str, int | None], int] = {}
    for doc in documents:
        for fragmento in split_text(doc.text, chunk_size, chunk_overlap):
            clave = (doc.source, doc.page)
            posicion = posiciones.get(clave, 0)
            posiciones[clave] = posicion + 1
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc.source, doc.page, posicion, fragmento),
                    text=fragmento,
                    source=doc.source,
                    page=doc.page,
                    doc_type=doc.doc_type,
                )
            )
    return chunks
