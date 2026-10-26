"""
Limpieza de ruido y normalización de texto (Unidad 3).

Implementa, como funciones puras y testeables, las cinco prácticas de
limpieza señaladas en el material del curso:

  1. Eliminar encabezados y pies de página repetidos
  2. Quitar etiquetas y marcado residual
  3. Normalizar espacios y saltos de línea
  4. Corregir la codificación de caracteres (encoding)
  5. Eliminar contenido duplicado
"""
import re
from collections import Counter

import ftfy

from ingestion.extractors import RawDocument

# --- 1. Encabezados/pies de página repetidos --------------------------------


def find_repeated_lines(documents: list[RawDocument], min_occurrences: int = 3) -> set[str]:
    """
    Detecta líneas que se repiten idénticas en múltiples páginas/documentos
    de la MISMA fuente: es la firma típica de un encabezado o pie de página
    (ej. "Diplomado en ... · Página 4"), que no aporta significado semántico.
    """
    contador: Counter[tuple[str, str]] = Counter()
    for doc in documents:
        lineas = {linea.strip() for linea in doc.text.splitlines() if linea.strip()}
        for linea in lineas:
            # Ignora líneas muy cortas (probablemente números de página sueltos,
            # que ya se tratan aparte) o muy largas (poco probable que sea un header).
            if 3 <= len(linea) <= 120:
                contador[(doc.source, linea)] += 1

    return {
        linea for (fuente, linea), veces in contador.items() if veces >= min_occurrences
    }


def remove_repeated_lines(documents: list[RawDocument]) -> list[RawDocument]:
    """Elimina, de cada documento, las líneas identificadas como encabezado/pie repetido."""
    lineas_ruido = find_repeated_lines(documents)
    limpios = []
    for doc in documents:
        nuevas_lineas = [
            linea for linea in doc.text.splitlines() if linea.strip() not in lineas_ruido
        ]
        limpios.append(
            RawDocument(
                source=doc.source,
                text="\n".join(nuevas_lineas),
                page=doc.page,
                doc_type=doc.doc_type,
            )
        )
    return limpios


# --- 2. Etiquetas y marcado residual -----------------------------------------

_RESIDUAL_TAG_RE = re.compile(r"<[^>]+>")
_MARKDOWN_SYMBOLS_RE = re.compile(r"[#*_`>~]{1,}")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def strip_residual_markup(text: str) -> str:
    """Elimina restos de etiquetas HTML/XML y caracteres de control que
    puedan haber quedado tras la extracción."""
    text = _RESIDUAL_TAG_RE.sub(" ", text)
    text = _CONTROL_CHARS_RE.sub("", text)
    return text


def strip_markdown_symbols(text: str) -> str:
    """Elimina símbolos de formato Markdown (#, *, _, `) manteniendo el texto."""
    return _MARKDOWN_SYMBOLS_RE.sub("", text)


# --- 3. Normalización de espacios y saltos de línea --------------------------

_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """Colapsa espacios múltiples y estandariza los saltos de línea."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    lineas = [linea.strip() for linea in text.split("\n")]
    return "\n".join(lineas).strip()


# --- 4. Codificación de caracteres -------------------------------------------


def fix_encoding(text: str) -> str:
    """Corrige errores comunes de encoding (tildes y símbolos mal
    interpretados), usando la librería open source `ftfy`."""
    return ftfy.fix_text(text)


# --- 5. Contenido duplicado ---------------------------------------------------


def deduplicate_paragraphs(text: str) -> str:
    """Elimina párrafos exactamente duplicados dentro de un mismo texto."""
    vistos: set[str] = set()
    parrafos_unicos = []
    for parrafo in text.split("\n\n"):
        clave = parrafo.strip()
        if clave and clave not in vistos:
            vistos.add(clave)
            parrafos_unicos.append(parrafo)
    return "\n\n".join(parrafos_unicos)


# --- Pipeline completo --------------------------------------------------------


def clean_document(doc: RawDocument) -> RawDocument:
    """Aplica, en orden, la limpieza a nivel de un solo documento
    (encoding → marcado residual → espacios → duplicados)."""
    texto = fix_encoding(doc.text)
    texto = strip_residual_markup(texto)
    if doc.doc_type == "markdown":
        texto = strip_markdown_symbols(texto)
    texto = normalize_whitespace(texto)
    texto = deduplicate_paragraphs(texto)

    return RawDocument(source=doc.source, text=texto, page=doc.page, doc_type=doc.doc_type)


def clean_corpus(documents: list[RawDocument]) -> list[RawDocument]:
    """Pipeline de limpieza completo sobre una colección de documentos:
    primero elimina encabezados/pies repetidos ENTRE documentos, y luego
    aplica la limpieza individual a cada uno."""
    sin_repetidos = remove_repeated_lines(documents)
    return [clean_document(doc) for doc in sin_repetidos if doc.text.strip()]
