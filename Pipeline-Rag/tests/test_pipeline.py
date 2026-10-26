import pytest

from ingestion import pipeline


@pytest.fixture
def docs_dir(tmp_path):
    (tmp_path / "a.md").write_text("# Política A\n\nEl plazo es de 30 días calendario.", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Política B\n\nLa garantía cubre 12 meses.", encoding="utf-8")
    return tmp_path


@pytest.fixture
def use_temp_store(temp_vector_store, monkeypatch):
    monkeypatch.setattr(pipeline, "VectorStore", lambda **kw: temp_vector_store)
    return temp_vector_store


def test_reingesting_does_not_duplicate_chunks(docs_dir, use_temp_store):
    primero = pipeline.run_pipeline(source_dir=str(docs_dir))
    segundo = pipeline.run_pipeline(source_dir=str(docs_dir))

    assert primero["total_en_coleccion"] == segundo["total_en_coleccion"] == primero["chunks_generados"]
    assert segundo["fuentes_sin_cambios"] == 2


def test_editing_a_document_replaces_its_chunks_only(docs_dir, use_temp_store):
    pipeline.run_pipeline(source_dir=str(docs_dir))
    (docs_dir / "a.md").write_text("# Política A\n\nEl plazo es de 45 días calendario.", encoding="utf-8")

    resumen = pipeline.run_pipeline(source_dir=str(docs_dir))

    textos = " ".join(r["text"] for r in use_temp_store.get_all() if r["source"] == "a.md")
    assert "45 días" in textos and "30 días" not in textos
    assert resumen["fuentes_sin_cambios"] == 1  # b.md no cambió


def test_reset_flag_starts_from_empty_collection(docs_dir, use_temp_store):
    pipeline.run_pipeline(source_dir=str(docs_dir))
    (docs_dir / "b.md").unlink()

    pipeline.run_pipeline(source_dir=str(docs_dir), reset=True)

    assert {r["source"] for r in use_temp_store.get_all()} == {"a.md"}
