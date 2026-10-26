from rag.chunking import Chunk
from rag.retriever import Retriever


def _make_chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id="1", text="El plazo de reembolso es de 30 días calendario.", source="politica.md", page=None, doc_type="markdown"),
        Chunk(chunk_id="2", text="La garantía cubre defectos de fábrica por 12 meses.", source="politica.md", page=None, doc_type="markdown"),
        Chunk(chunk_id="3", text="Reinicia el router desconectándolo 10 segundos.", source="soporte.html", page=None, doc_type="html"),
        Chunk(chunk_id="4", text="Los tickets se abren desde el portal web.", source="soporte.html", page=None, doc_type="html"),
    ]


def test_semantic_search_returns_top_k(temp_vector_store):
    temp_vector_store.add_chunks(_make_chunks())
    retriever = Retriever(temp_vector_store)

    resultados = retriever.semantic_search("¿Cómo soluciono un problema de wifi?", top_k=2)
    assert len(resultados) == 2


def test_keyword_search_finds_exact_word_match(temp_vector_store):
    temp_vector_store.add_chunks(_make_chunks())
    retriever = Retriever(temp_vector_store)

    resultados = retriever.keyword_search("reembolso plazo", top_k=4)
    assert len(resultados) > 0
    # El chunk que contiene literalmente "reembolso" y "plazo" debe quedar primero.
    assert "reembolso" in resultados[0].text.lower()


def test_keyword_search_on_empty_store_returns_empty_list(temp_vector_store):
    retriever = Retriever(temp_vector_store)
    assert retriever.keyword_search("algo", top_k=3) == []


def test_hybrid_search_combines_both_signals(temp_vector_store):
    temp_vector_store.add_chunks(_make_chunks())
    retriever = Retriever(temp_vector_store)

    resultados = retriever.hybrid_search("garantía de fábrica", top_k=3)
    assert 0 < len(resultados) <= 3
    assert all(r.score >= 0 for r in resultados)


def test_agentic_search_returns_rounds_trace(temp_vector_store):
    temp_vector_store.add_chunks(_make_chunks())
    retriever = Retriever(temp_vector_store)

    resultado = retriever.agentic_search("¿Cuánto dura la garantía?", top_k=2)

    assert isinstance(resultado.chunks, list)
    assert len(resultado.rounds) >= 1
    for ronda in resultado.rounds:
        assert {"round", "top_k", "avg_score", "n_results"} <= ronda.keys()


def test_agentic_search_stops_within_max_rounds(temp_vector_store):
    from config.settings import settings

    temp_vector_store.add_chunks(_make_chunks())
    retriever = Retriever(temp_vector_store)

    resultado = retriever.agentic_search("consulta cualquiera", top_k=2)
    assert len(resultado.rounds) <= settings.MAX_RETRIEVAL_ROUNDS


# --- P0-4: BM25 sin puntuación y sin índice obsoleto ----------------------------


def test_keyword_search_ignores_attached_punctuation_and_accents(temp_vector_store):
    temp_vector_store.add_chunks(_make_chunks())
    retriever = Retriever(temp_vector_store)

    # "garantia" sin tilde debe encontrar "garantía"; "reembolso." no debe depender del punto.
    assert "garantía" in retriever.keyword_search("garantia", top_k=1)[0].text
    assert "reembolso" in retriever.keyword_search("reembolso.", top_k=1)[0].text


def test_keyword_search_matches_exact_numbers(temp_vector_store):
    temp_vector_store.add_chunks([
        Chunk(chunk_id="a", text="La tasa del DPF a 360 días es 4,5% anual.", source="c.html", page=None, doc_type="html"),
        Chunk(chunk_id="b", text="La tasa de la caja de ahorro es 2,0% anual.", source="p.pdf", page=1, doc_type="pdf"),
    ])
    retriever = Retriever(temp_vector_store)

    assert retriever.keyword_search("tasa 4,5%", top_k=1)[0].chunk_id == "a"


def test_bm25_index_refreshes_after_corpus_changes(temp_vector_store):
    temp_vector_store.add_chunks(_make_chunks())
    retriever = Retriever(temp_vector_store)
    assert retriever.keyword_search("criptomonedas", top_k=3)[0].score == 0.0  # nada coincide

    temp_vector_store.add_chunks([
        Chunk(chunk_id="9", text="Aceptamos criptomonedas en agencias seleccionadas.", source="nuevo.md", page=None, doc_type="markdown"),
    ])
    assert retriever.keyword_search("criptomonedas", top_k=1)[0].chunk_id == "9"

    temp_vector_store.delete_source("nuevo.md")
    assert all(r.chunk_id != "9" for r in retriever.keyword_search("criptomonedas", top_k=4))


# --- P0-5: trazabilidad del método y de los puntajes parciales ---------------------


def test_keyword_results_expose_method_and_scores(temp_vector_store):
    temp_vector_store.add_chunks(_make_chunks())
    r = Retriever(temp_vector_store).keyword_search("reembolso plazo", top_k=1)[0]

    assert r.method == "bm25"
    assert r.chunk_id == "1" and r.doc_type == "markdown"
    assert r.scores["bm25"] == r.score and r.scores["bm25_raw"] > 0


def test_hybrid_results_expose_partial_scores(temp_vector_store):
    from config.settings import settings

    temp_vector_store.add_chunks(_make_chunks())
    resultados = Retriever(temp_vector_store).hybrid_search("garantía de fábrica", top_k=3)

    for r in resultados:
        assert r.method == "hybrid"
        assert r.chunk_id is not None
        assert set(r.scores) == {"semantic", "bm25", "hybrid"}
        esperado = settings.VECTOR_WEIGHT * (r.scores["semantic"] or 0.0) + settings.BM25_WEIGHT * (r.scores["bm25"] or 0.0)
        assert abs(r.score - esperado) < 1e-9

def test_retriever_accepts_custom_tokenizer(temp_vector_store):
    temp_vector_store.add_chunks(_make_chunks())
    legacy = Retriever(temp_vector_store, tokenizer=lambda s: s.lower().split())

    # Con la tokenización original la puntuación pegada impide la coincidencia.
    assert legacy.keyword_search("reembolso.", top_k=1)[0].score == 0.0
    assert Retriever(temp_vector_store).keyword_search("reembolso.", top_k=1)[0].score > 0.0

def test_hybrid_rrf_search_fuses_by_rank(temp_vector_store):
    temp_vector_store.add_chunks(_make_chunks())
    retriever = Retriever(temp_vector_store)

    resultados = retriever.hybrid_rrf_search("plazo de reembolso", top_k=3)
    assert 0 < len(resultados) <= 3
    assert all(r.method == "hybrid_rrf" and "rrf" in r.scores for r in resultados)
    assert [r.score for r in resultados] == sorted((r.score for r in resultados), reverse=True)
    assert "reembolso" in resultados[0].text.lower()
    # Máximo teórico: primer lugar en ambas listas
    assert resultados[0].score <= 2 / 61 + 1e-9
