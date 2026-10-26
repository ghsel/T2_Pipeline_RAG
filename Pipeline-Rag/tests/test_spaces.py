import tempfile

import pytest

from config.settings import settings
from rag.chunking import Chunk
from rag.retriever import Retriever
from rag.spaces import DEFAULT_SPACE, collection_name_for, delete_space, list_spaces, validate_space_name
from rag.vector_store import VectorStore


def _chunk(cid, text, source):
    return Chunk(chunk_id=cid, text=text, source=source, page=None, doc_type="markdown")


def test_default_space_keeps_original_collection():
    assert collection_name_for(None) == settings.COLLECTION_NAME
    assert collection_name_for(DEFAULT_SPACE) == settings.COLLECTION_NAME
    assert collection_name_for("Banco-Demo") == "space_banco-demo"


@pytest.mark.parametrize("malo", ["", "  ", "con espacio", "../x", "ñandú", "-inicio", "a" * 49])
def test_invalid_space_names_are_rejected(malo):
    with pytest.raises(ValueError):
        validate_space_name(malo)


def test_query_does_not_cross_spaces(fake_embeddings):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        a = VectorStore(persist_directory=d, space="alfa")
        b = VectorStore(persist_directory=d, space="beta")
        a.add_chunks([_chunk("a1", "El plazo de reembolso es de 30 días.", "a.md")])
        b.add_chunks([_chunk("b1", "La tasa del crédito de vivienda es 5,5%.", "b.md")])

        assert {r.source for r in a.query("plazo", top_k=5)} == {"a.md"}
        assert {r.source for r in b.query("plazo", top_k=5)} == {"b.md"}
        # También la búsqueda léxica (BM25) queda confinada al espacio
        assert {r.source for r in Retriever(b).keyword_search("reembolso plazo", top_k=5)} <= {"b.md"}
        assert Retriever(a).keyword_search("tasa vivienda", top_k=5) == [] or all(
            r.source == "a.md" for r in Retriever(a).keyword_search("tasa vivienda", top_k=5)
        )


def test_list_and_delete_spaces(fake_embeddings):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        VectorStore(persist_directory=d).add_chunks([_chunk("d1", "texto por defecto", "d.md")])
        s = VectorStore(persist_directory=d, space="tesis")
        s.add_chunks([_chunk("t1", "uno", "t.md"), _chunk("t2", "dos", "t.md")])

        espacios = {e["space"]: e["n_chunks"] for e in list_spaces(s.client)}
        assert espacios == {"default": 1, "tesis": 2}

        assert delete_space(s.client, "tesis") is True
        assert delete_space(s.client, "tesis") is False
        assert {e["space"] for e in list_spaces(s.client)} == {"default"}
