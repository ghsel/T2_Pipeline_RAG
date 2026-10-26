from rag.chunking import Chunk
from rag.graph import GraphIndex, extract_entities
from rag.retriever import Retriever


def _chunk(cid, text, source):
    return Chunk(chunk_id=cid, text=text, source=source, page=None, doc_type="markdown")


CORPUS = [
    _chunk("a", "El recargo por mora es de 0,05% diario sobre la cuota vencida.", "tarifario.md"),
    _chunk("b", "Un crédito entra en mora al día siguiente del vencimiento. Se aplica un recargo de 0,05% diario.", "politica.md"),
    _chunk("c", "Los créditos superiores a Bs 500.000 los aprueba el Comité Nacional de Créditos.", "manual.md"),
    _chunk("d", "La reprogramación la aprueba la misma instancia que aprobó el crédito original.", "politica2.md"),
    _chunk("e", "La caja de ahorro en dólares paga 0,10% anual.", "captaciones.md"),
    _chunk("f", "Horario de agencias: lunes a viernes.", "faq.md"),
]


def test_extract_entities_by_rules():
    ents = extract_entities("El Crédito PYME de Bs 800.000 a 6,5% pasa al Comité Nacional.")
    assert ents["Crédito PYME"] == "producto"
    assert ents["Comité Nacional de Créditos"] == "instancia"
    assert ents["% 6,5"] == "valor" and ents["BS 800.000"] == "valor"
    assert extract_entities("Texto sin nada relevante") == {}


def test_graph_links_chunks_through_shared_values_with_evidence():
    g = GraphIndex([{"chunk_id": c.chunk_id, "text": c.text, "source": c.source, "page": None} for c in CORPUS])
    assert g.graph.has_edge("Mora", "% 0,05")
    assert set(g.graph["Mora"]["% 0,05"]["evidence"]) == {"a", "b"}


def test_expand_adds_unseen_related_chunk_and_never_a_seed():
    g = GraphIndex([{"chunk_id": c.chunk_id, "text": c.text, "source": c.source, "page": None} for c in CORPUS])
    añadidos, traza = g.expand("¿Qué recargo se aplica por mora?", seed_chunk_ids=["a"], extra=2, hub_ratio=0.9)
    ids = [c for c, _ in añadidos]
    assert "b" in ids and "a" not in ids
    assert traza.added[0]["source"] and traza.query_entities


def test_expand_is_bounded_and_empty_without_entities():
    g = GraphIndex([{"chunk_id": c.chunk_id, "text": c.text, "source": c.source, "page": None} for c in CORPUS])
    assert g.expand("hola", seed_chunk_ids=["f"], extra=2)[0] == []
    añadidos, traza = g.expand("mora", ["a"], extra=1, max_nodes=2, hub_ratio=0.9)
    assert len(añadidos) <= 1 and len(traza.reached) <= 2


def test_persistence_is_versioned(tmp_path):
    reg = [{"chunk_id": c.chunk_id, "text": c.text, "source": c.source, "page": None} for c in CORPUS]
    GraphIndex.load_or_build(reg, "v1", tmp_path, "col")
    assert (tmp_path / "col.json").exists()
    # misma versión: se reutiliza lo persistido aunque se entreguen otros registros
    assert len(GraphIndex.load_or_build([], "v1", tmp_path, "col").records) == len(reg)
    # versión distinta: se reconstruye
    assert len(GraphIndex.load_or_build(reg[:2], "v2", tmp_path, "col").records) == 2


def test_graph_search_keeps_context_size_and_marks_graph_chunks(temp_vector_store):
    temp_vector_store.add_chunks(CORPUS)
    r = Retriever(temp_vector_store)
    res = r.graph_search("¿Qué recargo se aplica por mora?", top_k=4, extra=2)
    assert len(res) <= 4
    assert len({c.chunk_id for c in res}) == len(res)  # sin duplicados
    assert r.last_graph_trace is not None
    assert all(c.method in ("hybrid", "graph_expansion") for c in res)


def test_graph_search_with_extra_zero_equals_hybrid(temp_vector_store):
    temp_vector_store.add_chunks(CORPUS)
    r = Retriever(temp_vector_store)
    assert [c.chunk_id for c in r.graph_search("mora", 4, extra=0)] == [c.chunk_id for c in r.hybrid_search("mora", 4)]


def test_graph_rebuilds_when_corpus_changes(temp_vector_store):
    temp_vector_store.add_chunks(CORPUS[:2])
    r = Retriever(temp_vector_store)
    v1 = r.get_graph().version
    temp_vector_store.add_chunks(CORPUS[2:])
    assert r.get_graph().version != v1 and len(r.get_graph().records) == len(CORPUS)


def test_search_uses_graph_only_when_enabled(temp_vector_store):
    temp_vector_store.add_chunks(CORPUS)
    off = Retriever(temp_vector_store, use_graph=False)
    on = Retriever(temp_vector_store, use_graph=True)
    assert all(c.method != "graph_expansion" for c in off.search("¿Qué recargo se aplica por mora?", 4))
    assert on.search("¿Qué recargo se aplica por mora?", 4) and on.last_graph_trace is not None
    assert off.last_graph_trace is None
