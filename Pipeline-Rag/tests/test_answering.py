import pytest

from rag import answering
from rag.answering import answer_question, declares_no_evidence
from rag.augmentation import build_prompt
from rag.chunking import Chunk
from rag.evidence import EvidenceAssessment, EvidenceStatus
from rag.generator import OllamaUnavailableError
from rag.retriever import Retriever


def _chunks():
    return [
        Chunk(chunk_id="1", text="La tasa del crédito de vivienda es 5,5% anual.", source="01.md", page=None, doc_type="markdown"),
        Chunk(chunk_id="2", text="El plazo máximo es de 25 años para compra.", source="01.md", page=None, doc_type="markdown"),
    ]


@pytest.fixture
def retriever(temp_vector_store):
    temp_vector_store.add_chunks(_chunks())
    return Retriever(temp_vector_store)


def _force_evidence(monkeypatch, status, score=0.9):
    monkeypatch.setattr(answering, "assess_evidence",
                        lambda q, ch, **kw: EvidenceAssessment(status, score, {}, "forzado", "test", (0.3, 0.5)))


class Spy:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def __call__(self, prompt):
        self.calls += 1
        self.prompt = prompt
        return self.text


def test_insufficient_evidence_does_not_call_the_llm(retriever, monkeypatch):
    _force_evidence(monkeypatch, EvidenceStatus.INSUFFICIENT, 0.1)
    gen = Spy("no debería usarse")

    r = answer_question("¿capital de Bolivia?", retriever, agentic=False, use_policy=True, generate=gen)

    assert r.status == "sin_evidencia" and gen.calls == 0 and not r.llm_called
    assert "No encontré evidencia suficiente" in r.answer


def test_policy_disabled_still_calls_the_llm(retriever, monkeypatch):
    _force_evidence(monkeypatch, EvidenceStatus.INSUFFICIENT, 0.1)
    gen = Spy("La tasa es 5,5% [1].")

    r = answer_question("tasa vivienda", retriever, agentic=False, use_policy=False, prompt_version="v2", generate=gen)

    assert gen.calls == 1 and r.llm_called
    assert r.evidence.status is EvidenceStatus.INSUFFICIENT  # se calcula igual, solo no bloquea


def test_sufficient_evidence_with_valid_citations_is_respaldada(retriever, monkeypatch):
    _force_evidence(monkeypatch, EvidenceStatus.SUFFICIENT)
    r = answer_question("tasa vivienda", retriever, agentic=False, use_policy=True, prompt_version="v2",
                        generate=Spy("La tasa es 5,5% anual [1]."))
    assert r.status == "respaldada" and r.citations.has_valid_citation


def test_invalid_citation_downgrades_to_parcial_and_is_removed(retriever, monkeypatch):
    _force_evidence(monkeypatch, EvidenceStatus.SUFFICIENT)
    r = answer_question("tasa vivienda", retriever, agentic=False, use_policy=True, prompt_version="v2",
                        generate=Spy("La tasa es 5,5% [1][9]."))
    assert r.status == "parcial" and "[9]" not in r.answer and "[1]" in r.answer


def test_v2_without_citations_is_parcial(retriever, monkeypatch):
    _force_evidence(monkeypatch, EvidenceStatus.SUFFICIENT)
    r = answer_question("tasa vivienda", retriever, agentic=False, use_policy=True, prompt_version="v2",
                        generate=Spy("La tasa es 5,5%."))
    assert r.status == "parcial"


def test_v1_prompt_does_not_require_citations(retriever, monkeypatch):
    _force_evidence(monkeypatch, EvidenceStatus.SUFFICIENT)
    r = answer_question("tasa vivienda", retriever, agentic=False, use_policy=True, prompt_version="v1",
                        generate=Spy("La tasa es 5,5%."))
    assert r.status == "respaldada"


def test_partial_evidence_gives_parcial(retriever, monkeypatch):
    _force_evidence(monkeypatch, EvidenceStatus.PARTIAL, 0.4)
    r = answer_question("tasa vivienda", retriever, agentic=False, use_policy=True, prompt_version="v2",
                        generate=Spy("La tasa es 5,5% [1]."))
    assert r.status == "parcial"


def test_model_declining_is_sin_evidencia(retriever, monkeypatch):
    _force_evidence(monkeypatch, EvidenceStatus.SUFFICIENT)
    r = answer_question("tasa vivienda", retriever, agentic=False, use_policy=True, prompt_version="v2",
                        generate=Spy("No se menciona esa información en el contexto."))
    assert r.status == "sin_evidencia"


def test_generator_failure_is_error_with_chunks_preserved(retriever, monkeypatch):
    _force_evidence(monkeypatch, EvidenceStatus.SUFFICIENT)

    def broken(prompt):
        raise OllamaUnavailableError("Ollama no responde")

    r = answer_question("tasa vivienda", retriever, agentic=False, use_policy=True, generate=broken)
    assert r.status == "error" and "Ollama no responde" in r.error and r.chunks


def test_empty_store_is_sin_documentos(temp_vector_store):
    gen = Spy("x")
    r = answer_question("cualquier cosa", Retriever(temp_vector_store), agentic=False, generate=gen)
    assert r.status == "sin_documentos" and gen.calls == 0


def test_prompt_v2_numbers_the_fragments_and_asks_for_citations(retriever, monkeypatch):
    _force_evidence(monkeypatch, EvidenceStatus.SUFFICIENT)
    gen = Spy("ok [1]")
    answer_question("tasa vivienda", retriever, agentic=False, prompt_version="v2", generate=gen)
    assert "[1] Fuente:" in gen.prompt and "entre corchetes" in gen.prompt


def test_prompt_v1_is_unchanged_default():
    from rag.vector_store import RetrievedChunk
    p = build_prompt("¿x?", [RetrievedChunk(text="t", source="a.md", page=None, score=0.5)])
    assert "[Fragmento 1 — a.md | relevancia=0.50]" in p and "entre corchetes" not in p
    with pytest.raises(ValueError):
        build_prompt("¿x?", [], version="v9")


def test_declares_no_evidence():
    assert declares_no_evidence("Lo siento, no se encuentra información en el contexto.")
    assert not declares_no_evidence("La tasa es de 5,5% anual.")
