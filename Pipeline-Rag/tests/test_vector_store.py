from rag.chunking import Chunk


def _make_chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id="1", text="El plazo de reembolso es de 30 días.", source="politica.md", page=None, doc_type="markdown"),
        Chunk(chunk_id="2", text="La garantía cubre defectos de fábrica.", source="politica.md", page=None, doc_type="markdown"),
        Chunk(chunk_id="3", text="Reinicia el router 10 segundos.", source="soporte.html", page=None, doc_type="html"),
    ]


def test_add_and_count(temp_vector_store):
    assert temp_vector_store.count() == 0
    temp_vector_store.add_chunks(_make_chunks())
    assert temp_vector_store.count() == 3


def test_query_returns_expected_structure(temp_vector_store):
    temp_vector_store.add_chunks(_make_chunks())
    resultados = temp_vector_store.query("¿Cuál es el plazo de reembolso?", top_k=2)

    assert len(resultados) == 2
    for r in resultados:
        assert hasattr(r, "text")
        assert hasattr(r, "source")
        assert hasattr(r, "score")
        assert 0.0 <= r.score <= 1.0


def test_query_on_empty_store_returns_empty_list(temp_vector_store):
    assert temp_vector_store.query("cualquier cosa", top_k=3) == []


def test_get_all_preserves_metadata(temp_vector_store):
    temp_vector_store.add_chunks(_make_chunks())
    registros = temp_vector_store.get_all()

    assert len(registros) == 3
    fuentes = {r["source"] for r in registros}
    assert fuentes == {"politica.md", "soporte.html"}


def test_reset_clears_the_collection(temp_vector_store):
    temp_vector_store.add_chunks(_make_chunks())
    assert temp_vector_store.count() == 3

    temp_vector_store.reset()
    assert temp_vector_store.count() == 0


# --- P0-3: idempotencia y gestión por documento --------------------------------


def test_add_chunks_twice_does_not_duplicate(temp_vector_store):
    temp_vector_store.add_chunks(_make_chunks())
    temp_vector_store.add_chunks(_make_chunks())
    assert temp_vector_store.count() == 3


def test_delete_source_removes_only_that_source(temp_vector_store):
    temp_vector_store.add_chunks(_make_chunks())

    borrados = temp_vector_store.delete_source("politica.md")

    assert borrados == 2
    assert temp_vector_store.count() == 1
    assert {r["source"] for r in temp_vector_store.get_all()} == {"soporte.html"}
    assert temp_vector_store.delete_source("no_existe.md") == 0


def test_replace_source_removes_obsolete_chunks_and_adds_new(temp_vector_store):
    temp_vector_store.add_chunks(_make_chunks())
    nuevos = [
        Chunk(chunk_id="1", text="El plazo de reembolso es de 30 días.", source="politica.md", page=None, doc_type="markdown"),
        Chunk(chunk_id="9", text="Texto nuevo de la política.", source="politica.md", page=None, doc_type="markdown"),
    ]

    resultado = temp_vector_store.replace_source("politica.md", nuevos)

    assert resultado == {"source": "politica.md", "deleted": 1, "added": 1, "unchanged": False}
    textos = {r["text"] for r in temp_vector_store.get_all() if r["source"] == "politica.md"}
    assert textos == {"El plazo de reembolso es de 30 días.", "Texto nuevo de la política."}
    assert temp_vector_store.count() == 3  # soporte.html no se tocó


def test_replace_source_unchanged_skips_work(temp_vector_store):
    temp_vector_store.add_chunks(_make_chunks())
    mismos = [c for c in _make_chunks() if c.source == "politica.md"]

    resultado = temp_vector_store.replace_source("politica.md", mismos)

    assert resultado["unchanged"] is True
    assert temp_vector_store.count() == 3


def test_list_sources_summarizes_documents(temp_vector_store):
    assert temp_vector_store.list_sources() == []
    temp_vector_store.add_chunks(_make_chunks())

    assert temp_vector_store.list_sources() == [
        {"source": "politica.md", "doc_type": "markdown", "n_chunks": 2},
        {"source": "soporte.html", "doc_type": "html", "n_chunks": 1},
    ]


def test_index_version_changes_with_content(temp_vector_store):
    v_vacio = temp_vector_store.index_version()
    temp_vector_store.add_chunks(_make_chunks())
    v_lleno = temp_vector_store.index_version()
    temp_vector_store.add_chunks(_make_chunks())  # mismo contenido
    assert v_lleno == temp_vector_store.index_version()
    temp_vector_store.delete_source("soporte.html")

    assert v_vacio != v_lleno != temp_vector_store.index_version()


# --- P0-5: trazabilidad -----------------------------------------------------------


def test_query_exposes_traceability_fields(temp_vector_store):
    temp_vector_store.add_chunks(_make_chunks())

    r = temp_vector_store.query("plazo de reembolso", top_k=3)[0]

    assert r.chunk_id in {"1", "2", "3"}
    assert r.doc_type in {"markdown", "html"}
    assert r.method == "semantic"
    assert r.scores["semantic"] == r.score


def test_get_all_includes_chunk_id_and_doc_type(temp_vector_store):
    temp_vector_store.add_chunks(_make_chunks())
    registros = {r["chunk_id"]: r for r in temp_vector_store.get_all()}
    assert registros["3"]["doc_type"] == "html"