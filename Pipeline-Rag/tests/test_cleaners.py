from ingestion.cleaners import (
    clean_corpus,
    deduplicate_paragraphs,
    fix_encoding,
    normalize_whitespace,
    remove_repeated_lines,
    strip_residual_markup,
)
from ingestion.extractors import RawDocument


def test_fix_encoding_repairs_mojibake():
    texto_roto = "PolÃ­tica de reembolsos"  # mojibake típico de UTF-8 mal decodificado
    assert fix_encoding(texto_roto) == "Política de reembolsos"


def test_strip_residual_markup_removes_html_tags():
    texto = "Texto <b>importante</b> con <span class='x'>marcado</span> residual."
    resultado = strip_residual_markup(texto)
    assert "<b>" not in resultado
    assert "<span" not in resultado
    assert "importante" in resultado


def test_normalize_whitespace_collapses_spaces_and_newlines():
    texto = "Hola    mundo\n\n\n\ncon   espacios\t\textra  "
    resultado = normalize_whitespace(texto)
    assert "    " not in resultado
    assert "\n\n\n" not in resultado
    assert resultado == "Hola mundo\n\ncon espacios extra"


def test_deduplicate_paragraphs_removes_exact_duplicates():
    texto = "Párrafo A.\n\nPárrafo B.\n\nPárrafo A.\n\nPárrafo C."
    resultado = deduplicate_paragraphs(texto)
    assert resultado.count("Párrafo A.") == 1
    assert "Párrafo B." in resultado
    assert "Párrafo C." in resultado


def test_remove_repeated_lines_strips_headers_and_footers():
    encabezado = "Diplomado en IA Generativa · Módulo 2"
    documentos = [
        RawDocument(source="a.pdf", text=f"{encabezado}\nContenido único página 1", page=1, doc_type="pdf"),
        RawDocument(source="a.pdf", text=f"{encabezado}\nContenido único página 2", page=2, doc_type="pdf"),
        RawDocument(source="a.pdf", text=f"{encabezado}\nContenido único página 3", page=3, doc_type="pdf"),
    ]

    limpios = remove_repeated_lines(documentos)

    for doc in limpios:
        assert encabezado not in doc.text
    assert "Contenido único página 1" in limpios[0].text
    assert "Contenido único página 2" in limpios[1].text


def test_clean_corpus_end_to_end():
    documentos = [
        RawDocument(
            source="doc.html",
            text="  Encabezado repetido\nInformaciÃ³n <b>importante</b>   con   ruido.\n\nEncabezado repetido\nMás info.  ",
            doc_type="html",
        ),
        RawDocument(
            source="doc.html",
            text="  Encabezado repetido\nOtra secciÃ³n relevante.  ",
            doc_type="html",
        ),
    ]

    resultado = clean_corpus(documentos)

    assert len(resultado) == 2
    texto_completo = " ".join(d.text for d in resultado)
    assert "<b>" not in texto_completo
    assert "Información" in texto_completo
    assert "sección" in texto_completo.lower() or "Sección" in texto_completo
