from rag.reranker import rerank
from rag.retriever import Retriever
from rag.vector_store import RetrievedChunk


def _c(cid, text, score=0.5):
    return RetrievedChunk(text=text, source="s.md", page=None, score=score, chunk_id=cid, method="hybrid", scores={"hybrid": score})


def test_rerank_orders_by_model_score_and_keeps_traceability():
    chunks = [_c("a", "irrelevante"), _c("b", "plazo de reembolso"), _c("c", "otro")]
    predict = lambda q, t: 1.0 if "reembolso" in t else 0.1  # noqa: E731
    res = rerank("reembolso", chunks, top_k=2, model_name="x", predict=predict)
    assert [r.chunk_id for r in res] == ["b", "a"]  # empate -> conserva el orden previo
    assert res[0].method == "hybrid_rerank" and res[0].scores["pre_rank"] == 2 and res[0].scores["rerank"] == 1.0
    assert res[0].scores["hybrid"] == 0.5  # conserva puntajes previos


def test_rerank_handles_empty_and_top_k_larger_than_candidates():
    assert rerank("q", [], 4, "x", predict=lambda q, t: 0) == []
    assert len(rerank("q", [_c("a", "t")], 4, "x", predict=lambda q, t: 0)) == 1


def test_rerank_search_uses_wider_candidate_pool_than_top_k(temp_vector_store, monkeypatch):
    from rag.chunking import Chunk
    temp_vector_store.add_chunks([Chunk(chunk_id=str(i), text=f"documento {i} sobre créditos", source="d.md", page=None, doc_type="markdown") for i in range(6)])
    vistos = {}

    def falso(query, chunks, top_k, model_name, predict=None):
        vistos.update(n=len(chunks), model=model_name)
        return chunks[:top_k]

    monkeypatch.setattr("rag.retriever.rerank", falso)
    res = Retriever(temp_vector_store).rerank_search("créditos", top_k=2, candidates=5, model="m")
    assert vistos == {"n": 5, "model": "m"} and len(res) == 2


def test_search_prefers_rerank_when_enabled(temp_vector_store, monkeypatch):
    temp_vector_store.add_chunks([])
    llamado = []
    monkeypatch.setattr(Retriever, "rerank_search", lambda self, q, k, **kw: llamado.append(q) or [])
    Retriever(temp_vector_store, use_rerank=True).search("hola", 4)
    Retriever(temp_vector_store, use_rerank=False).search("hola", 4)
    assert llamado == ["hola"]
