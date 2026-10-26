from rag.answering import answer_question
from rag.cache import ExactCache, normalize_query
from rag.chunking import Chunk
from rag.generator import OllamaUnavailableError
from rag.retriever import Retriever


def _chunk(cid, text, source="doc.md"):
    return Chunk(chunk_id=cid, text=text, source=source, page=None, doc_type="markdown")


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        return f"Respuesta {self.calls} [1]"


def _setup(store):
    store.add_chunks([_chunk("c1", "El plazo de reembolso es de 30 días calendario."),
                      _chunk("c2", "La garantía cubre defectos de fábrica por 12 meses.")])
    store.cache = ExactCache()
    return Retriever(store), store.cache


def test_normalize_query_only_collapses_whitespace():
    assert normalize_query("  ¿Cuál   es el plazo? ") == "¿Cuál es el plazo?"
    assert normalize_query("Plazo") != normalize_query("plazo")  # caché exacta, no semántica


def test_lru_evicts_oldest_entry():
    c = ExactCache(max_entries=2)
    c.put("answer", "a", 1); c.put("answer", "b", 2); c.get("answer", "a"); c.put("answer", "c", 3)
    assert c.get("answer", "b") is None and c.get("answer", "a") == 1 and c.get("answer", "c") == 3


def test_stored_values_cannot_be_mutated_by_callers():
    c = ExactCache()
    c.put("retrieval", "k", {"x": [1]})
    c.get("retrieval", "k")["x"].append(2)
    assert c.get("retrieval", "k") == {"x": [1]}


def test_answer_layer_skips_llm_on_repeat(temp_vector_store):
    retriever, cache = _setup(temp_vector_store)
    llm = FakeLLM()
    kw = dict(top_k=2, agentic=False, use_policy=False, prompt_version="v2", generate=llm, cache=cache)

    primera = answer_question("plazo de reembolso", retriever, **kw)
    segunda = answer_question("  plazo de  reembolso ", retriever, **kw)

    assert llm.calls == 1
    assert primera.cache_hits == [] and primera.llm_called
    assert segunda.cache_hits == ["answer"] and not segunda.llm_called
    assert segunda.answer == primera.answer and segunda.timings["generation_s"] == 0.0


def test_changing_prompt_version_reuses_retrieval_but_not_answer(temp_vector_store):
    retriever, cache = _setup(temp_vector_store)
    llm = FakeLLM()
    kw = dict(top_k=2, agentic=False, use_policy=False, generate=llm, cache=cache)

    answer_question("plazo de reembolso", retriever, prompt_version="v1", **kw)
    r = answer_question("plazo de reembolso", retriever, prompt_version="v2", **kw)

    assert llm.calls == 2
    assert r.cache_hits == ["retrieval"]


def test_changing_the_index_invalidates_cached_answers(temp_vector_store):
    retriever, cache = _setup(temp_vector_store)
    llm = FakeLLM()
    kw = dict(top_k=2, agentic=False, use_policy=False, prompt_version="v2", generate=llm, cache=cache)

    answer_question("plazo de reembolso", retriever, **kw)
    temp_vector_store.add_chunks([_chunk("c3", "Nuevo comunicado: el plazo de reembolso ahora es de 15 días.", "nuevo.md")])
    r = answer_question("plazo de reembolso", retriever, **kw)

    assert llm.calls == 2 and r.cache_hits == []
    assert "nuevo.md" in {c.source for c in r.chunks}  # no se sirvió la recuperación obsoleta


def test_errors_are_not_cached(temp_vector_store):
    retriever, cache = _setup(temp_vector_store)

    def caido(prompt):
        raise OllamaUnavailableError("sin Ollama")

    kw = dict(top_k=2, agentic=False, use_policy=False, prompt_version="v2", cache=cache)
    assert answer_question("plazo de reembolso", retriever, generate=caido, **kw).status == "error"
    llm = FakeLLM()
    r = answer_question("plazo de reembolso", retriever, generate=llm, **kw)
    assert llm.calls == 1 and r.status != "error"


def test_without_cache_behaviour_is_unchanged(temp_vector_store):
    retriever, _ = _setup(temp_vector_store)
    temp_vector_store.cache = None
    llm = FakeLLM()
    kw = dict(top_k=2, agentic=False, use_policy=False, prompt_version="v2", generate=llm)
    answer_question("plazo de reembolso", retriever, **kw)
    r = answer_question("plazo de reembolso", retriever, **kw)
    assert llm.calls == 2 and r.cache_hits == []


def test_embedding_layer_hits_on_repeated_query(temp_vector_store):
    retriever, cache = _setup(temp_vector_store)
    retriever.semantic_search("plazo", top_k=1)
    retriever.semantic_search("plazo", top_k=2)
    st = cache.stats()["embedding"]
    assert st["hits"] == 1 and st["misses"] == 1
