from ingestion.extractors import RawDocument
from rag.chunking import chunk_documents, split_text


def test_split_text_respects_chunk_size_roughly():
    texto = "Frase corta. " * 200  # ~2600 caracteres
    fragmentos = split_text(texto, chunk_size=300, chunk_overlap=0)

    assert len(fragmentos) > 1
    # Ningún fragmento debería exceder demasiado el tamaño pedido.
    assert all(len(f) <= 320 for f in fragmentos)


def test_split_text_short_text_returns_single_chunk():
    texto = "Un texto corto que cabe en un solo chunk."
    fragmentos = split_text(texto, chunk_size=500, chunk_overlap=50)
    assert fragmentos == [texto]


def test_split_text_empty_text_returns_no_chunks():
    assert split_text("", chunk_size=500) == []
    assert split_text("   ", chunk_size=500) == []


def test_split_text_prefers_paragraph_boundaries():
    parrafo_a = "A" * 100
    parrafo_b = "B" * 100
    texto = f"{parrafo_a}\n\n{parrafo_b}"

    fragmentos = split_text(texto, chunk_size=110, chunk_overlap=0)

    # Cada párrafo debería quedar en su propio chunk, sin mezclarse a la mitad.
    assert any(parrafo_a in f for f in fragmentos)
    assert any(parrafo_b in f for f in fragmentos)


def test_split_text_overlap_shares_content_between_chunks():
    texto = ("Oración número uno. " * 10) + ("Oración número dos. " * 10)
    fragmentos_sin_overlap = split_text(texto, chunk_size=150, chunk_overlap=0)
    fragmentos_con_overlap = split_text(texto, chunk_size=150, chunk_overlap=40)

    assert len(fragmentos_con_overlap) >= len(fragmentos_sin_overlap)
    # El segundo chunk con overlap debe contener el final del primero.
    cola_primer_chunk_base = fragmentos_sin_overlap[0][-30:]
    assert cola_primer_chunk_base in fragmentos_con_overlap[1]


def test_chunk_documents_preserves_source_metadata():
    documentos = [
        RawDocument(source="politica.md", text="Contenido de la política " * 30, doc_type="markdown"),
        RawDocument(source="manual.pdf", text="Contenido del manual " * 30, page=3, doc_type="pdf"),
    ]

    chunks = chunk_documents(documentos, chunk_size=200, chunk_overlap=20)

    fuentes = {c.source for c in chunks}
    assert fuentes == {"politica.md", "manual.pdf"}

    chunks_pdf = [c for c in chunks if c.source == "manual.pdf"]
    assert all(c.page == 3 for c in chunks_pdf)

    # Cada chunk debe tener un ID único.
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_ids_are_deterministic_across_runs():
    documentos = [RawDocument(source="politica.md", text="Contenido de la política " * 30, doc_type="markdown")]
    ids_1 = [c.chunk_id for c in chunk_documents(documentos, chunk_size=200, chunk_overlap=20)]
    ids_2 = [c.chunk_id for c in chunk_documents(documentos, chunk_size=200, chunk_overlap=20)]
    assert ids_1 == ids_2


def test_chunk_ids_are_unique_even_for_repeated_text():
    # Dos páginas con exactamente el mismo texto no deben colisionar.
    documentos = [
        RawDocument(source="manual.pdf", text="Texto repetido.", page=1, doc_type="pdf"),
        RawDocument(source="manual.pdf", text="Texto repetido.", page=2, doc_type="pdf"),
    ]
    ids = [c.chunk_id for c in chunk_documents(documentos)]
    assert len(ids) == len(set(ids)) == 2


def test_chunk_id_changes_when_text_changes():
    a = chunk_documents([RawDocument(source="d.md", text="Plazo de 30 días.", doc_type="markdown")])
    b = chunk_documents([RawDocument(source="d.md", text="Plazo de 45 días.", doc_type="markdown")])
    assert a[0].chunk_id != b[0].chunk_id
