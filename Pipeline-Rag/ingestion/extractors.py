"""
Extractores de texto para las tres fuentes cubiertas en la Unidad 3:
PDF, HTML y Markdown.

Cada extractor devuelve una lista de `RawDocument`: un fragmento de texto
crudo junto con su metadata de origen (archivo y, si aplica, número de
página). Esta metadata es la que luego permite ofrecer "citas verificables"
en la respuesta final del sistema RAG (ver Unidad 1, slide "Generation").

Retos abordados por fuente (según el material del curso):
  - PDF:      tablas, columnas y maquetación pueden romper el orden del
              texto → se extrae página por página y se conserva el número
              de página como metadata.
  - HTML:     hay que separar el contenido útil del ruido (menús, scripts,
              anuncios, etiquetas) → se eliminan explícitamente esas
              etiquetas antes de extraer el texto.
  - Markdown: ya tiene estructura semántica (encabezados, listas) → la
              extracción es casi directa, solo se limpian imágenes/enlaces.
"""
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

# Etiquetas HTML que casi nunca contienen contenido útil para una base de
# conocimiento (menús de navegación, scripts, anuncios, pies de página).
HTML_NOISE_TAGS = ["script", "style", "nav", "footer", "header", "noscript", "aside", "form"]


@dataclass
class RawDocument:
    """Un fragmento de texto crudo extraído de una fuente, con su metadata."""
    source: str          # nombre del archivo de origen
    text: str            # texto extraído (aún sin limpiar)
    page: int | None = None  # número de página (solo aplica a PDF)
    doc_type: str = "unknown"  # "pdf" | "html" | "markdown"


def extract_pdf(path: Path) -> list[RawDocument]:
    """Extrae texto de un PDF, página por página, preservando el número de página."""
    reader = PdfReader(str(path))
    documentos = []
    for i, page in enumerate(reader.pages, start=1):
        texto = page.extract_text() or ""
        if texto.strip():
            documentos.append(
                RawDocument(source=path.name, text=texto, page=i, doc_type="pdf")
            )
    return documentos


def extract_html(path: Path) -> list[RawDocument]:
    """Extrae el contenido útil de un HTML, descartando menús, scripts y anuncios."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    for tag_name in HTML_NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Preferimos el contenido de <main> si existe; si no, todo el <body>.
    contenedor = soup.find("main") or soup.body or soup
    texto = contenedor.get_text(separator="\n", strip=True)

    return [RawDocument(source=path.name, text=texto, doc_type="html")]


def extract_markdown(path: Path) -> list[RawDocument]:
    """Extrae texto de un Markdown. La estructura semántica ya existe,
    solo se eliminan imágenes y se simplifican enlaces."""
    texto = path.read_text(encoding="utf-8", errors="replace")

    # Elimina imágenes: ![alt](url)
    texto = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", texto)
    # Convierte enlaces [texto](url) -> texto (se conserva el texto, se descarta la URL)
    texto = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", texto)

    return [RawDocument(source=path.name, text=texto, doc_type="markdown")]


EXTRACTORS = {
    ".pdf": extract_pdf,
    ".html": extract_html,
    ".htm": extract_html,
    ".md": extract_markdown,
    ".markdown": extract_markdown,
}


def extract_document(path: Path) -> list[RawDocument]:
    """Despacha al extractor correcto según la extensión del archivo."""
    extractor = EXTRACTORS.get(path.suffix.lower())
    if extractor is None:
        raise ValueError(
            f"Formato no soportado: '{path.suffix}'. "
            f"Formatos válidos: {list(EXTRACTORS.keys())}"
        )
    return extractor(path)


def extract_directory(directory: Path) -> list[RawDocument]:
    """Extrae todos los documentos soportados dentro de un directorio."""
    documentos: list[RawDocument] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in EXTRACTORS:
            documentos.extend(extract_document(path))
    return documentos
