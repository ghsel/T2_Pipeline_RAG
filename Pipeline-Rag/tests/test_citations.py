from rag.citations import validate_citations
from rag.vector_store import RetrievedChunk


def _chunks():
    return [
        RetrievedChunk(text="2. Tasas de interés\nLa tasa es de 5,5% anual hasta UFV 255.000.", source="01_vivienda.md", page=None, score=0.9, chunk_id="c1"),
        RetrievedChunk(text="El plazo máximo es de 25 años para compra.", source="02_manual.pdf", page=3, score=0.8, chunk_id="c2"),
    ]


def test_valid_bracket_citations_are_resolved_to_chunks():
    r = validate_citations("La tasa es 5,5% [1] y el plazo 25 años [2].", _chunks())

    assert [c["n"] for c in r.citations] == [1, 2]
    assert all(c["valid"] for c in r.citations)
    assert r.cited_chunk_ids == ["c1", "c2"]
    assert r.citations[1]["source"] == "02_manual.pdf" and r.citations[1]["page"] == 3
    assert not r.has_errors and r.has_valid_citation


def test_out_of_range_citation_is_an_error_and_removed_from_text():
    r = validate_citations("La tasa es 5,5% [1][7].", _chunks())

    assert r.has_errors
    assert any(i["type"] == "invalid_citation" for i in r.issues)
    assert "[7]" not in r.text_clean
    assert "[1]" in r.text_clean


def test_grouped_and_fragment_style_markers():
    r = validate_citations("Ver [1, 2] y también [Fragmento 2].", _chunks())
    assert [c["n"] for c in r.citations] == [1, 2, 2]
    assert not r.has_errors


def test_free_text_fragment_mentions_are_checked():
    r = validate_citations("Según el Fragmento 9 la tasa es 5,5%.", _chunks())
    assert r.has_errors  # el original (v1) citaba así y puede inventar números


def test_fabricated_section_is_flagged_but_real_section_is_not():
    falsa = validate_citations("Según la sección 2.2 la tasa es 5,5% [1].", _chunks())
    assert any(i["type"] == "fabricated_reference" for i in falsa.issues)
    real = validate_citations("Según la sección 2 la tasa es 5,5% [1].", _chunks())
    assert not any(i["type"] == "fabricated_reference" for i in real.issues)


def test_file_not_retrieved_is_flagged():
    r = validate_citations("Consulte tarifario.pdf para más detalles [1].", _chunks())
    assert any("tarifario.pdf" in i["detail"] for i in r.issues)
    ok = validate_citations("Consulte 01_vivienda.md [1].", _chunks())
    assert not any(i["type"] == "fabricated_reference" for i in ok.issues)


def test_unsupported_number_is_a_warning_not_an_error():
    r = validate_citations("La tasa es 7,5% [1].", _chunks())
    avisos = [i for i in r.issues if i["type"] == "unsupported_number"]
    assert avisos and avisos[0]["severity"] == "warning" and "7,5" in avisos[0]["detail"]
    assert not r.has_errors


def test_numbers_from_the_query_are_allowed():
    r = validate_citations("Para UFV 300.000 aplica 5,5% [1].", _chunks(), query="vivienda de UFV 300.000")
    assert not any(i["type"] == "unsupported_number" for i in r.issues)


def test_no_citations_is_reported():
    r = validate_citations("La tasa es 5,5%.", _chunks())
    assert any(i["type"] == "no_citations" for i in r.issues)
    assert not r.has_valid_citation


def test_list_markers_and_citation_numbers_are_not_treated_as_figures():
    r = validate_citations("1. Primero [1]. 2. Segundo [2].", _chunks())
    assert not any(i["type"] == "unsupported_number" for i in r.issues)
