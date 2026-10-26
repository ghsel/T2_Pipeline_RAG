import pytest

from rag.evidence import EvidenceStatus, assess_evidence, classify, compute_features
from rag.vector_store import RetrievedChunk


def _chunk(text, source="a.md"):
    return RetrievedChunk(text=text, source=source, page=None, score=0.5, chunk_id=text[:5])


def _embed_factory(cosine: float):
    """Embebedor falso: la consulta es [1, 0] y cada fragmento tiene el coseno indicado con ella."""
    def embed(texts):
        q = [1.0, 0.0]
        resto = [[cosine, (1 - cosine ** 2) ** 0.5] for _ in texts[1:]]
        return [q] + resto
    return embed


def test_classify_thresholds():
    assert classify(0.8, 0.3, 0.5) is EvidenceStatus.SUFFICIENT
    assert classify(0.5, 0.3, 0.5) is EvidenceStatus.SUFFICIENT  # el límite superior es inclusivo
    assert classify(0.4, 0.3, 0.5) is EvidenceStatus.PARTIAL
    assert classify(0.3, 0.3, 0.5) is EvidenceStatus.PARTIAL
    assert classify(0.29, 0.3, 0.5) is EvidenceStatus.INSUFFICIENT


def test_features_cosine_and_term_coverage():
    chunks = [_chunk("La tasa del crédito de vivienda es 5,5% anual"), _chunk("Texto sin relación")]
    f = compute_features("tasa crédito vivienda", chunks, embed=_embed_factory(0.6))

    assert f["max_cosine"] == pytest.approx(0.6, abs=1e-3)
    assert f["coverage_best"] == 1.0          # los 3 términos están en el primer fragmento
    assert f["coverage_union"] == 1.0
    assert f["n_chunks"] == 2


def test_partial_coverage_across_chunks():
    chunks = [_chunk("tasa de interés"), _chunk("crédito de consumo")]
    f = compute_features("tasa crédito vivienda", chunks, embed=_embed_factory(0.5))
    assert f["coverage_best"] == pytest.approx(1 / 3, abs=1e-3)  # ningún fragmento cubre más de 1 de 3 términos
    assert f["coverage_union"] == pytest.approx(2 / 3, abs=1e-3)


def test_empty_store_is_no_documents():
    a = assess_evidence("lo que sea", [], store_is_empty=True)
    assert a.status is EvidenceStatus.NO_DOCUMENTS


def test_no_chunks_is_insufficient():
    a = assess_evidence("lo que sea", [], embed=_embed_factory(0.0))
    assert a.status is EvidenceStatus.INSUFFICIENT and a.score == 0.0


def test_strong_evidence_is_sufficient_and_weak_is_insufficient():
    fuerte = assess_evidence("tasa vivienda", [_chunk("tasa vivienda 5,5%")], scorer="blend", low=0.3, high=0.5, embed=_embed_factory(0.8))
    debil = assess_evidence("tasa vivienda", [_chunk("otro tema distinto")], scorer="blend", low=0.3, high=0.5, embed=_embed_factory(0.1))

    assert fuerte.status is EvidenceStatus.SUFFICIENT
    assert debil.status is EvidenceStatus.INSUFFICIENT
    assert fuerte.score > debil.score
    assert "blend" in fuerte.reason and fuerte.features["n_chunks"] == 1


def test_scorer_choice_changes_the_score():
    chunks = [_chunk("tasa vivienda")]
    emb = _embed_factory(0.2)
    c = assess_evidence("tasa vivienda", chunks, scorer="cosine", low=0.1, high=0.9, embed=emb)
    v = assess_evidence("tasa vivienda", chunks, scorer="coverage", low=0.1, high=0.9, embed=emb)
    assert c.score == pytest.approx(0.2, abs=1e-3) and v.score == 1.0
