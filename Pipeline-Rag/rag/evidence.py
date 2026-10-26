"""
Política de evidencia (P1-1): decide en el BACKEND, con señales verificables, si lo recuperado
alcanza para responder, en lugar de delegar esa decisión al LLM.

Estados:
  SUFFICIENT    la evidencia recuperada respalda razonablemente la pregunta.
  PARTIAL       hay evidencia relacionada pero débil o incompleta.
  INSUFFICIENT  nada de lo recuperado respalda la pregunta (no se llama al LLM).
  NO_DOCUMENTS  la base de conocimiento está vacía.

Señales (se calculan sobre los chunks recuperados, sin importar qué método los devolvió):
  max_cosine / top1_cosine   similitud coseno entre la consulta y el fragmento (embeddings locales).
  coverage_best              fracción de términos de la consulta presentes en el mejor fragmento.
  coverage_union             fracción de términos de la consulta presentes en el conjunto recuperado.

`score` es un indicador heurístico en [0, 1] para ordenar la evidencia: NO es una probabilidad ni una
medida de confianza calibrada. Los umbrales se calibran con datos (ver la evaluación en ai-desktop/eval).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from config.settings import settings
from rag.embeddings import embed_texts
from rag.text import tokenize
from rag.vector_store import RetrievedChunk


class EvidenceStatus(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"
    NO_DOCUMENTS = "NO_DOCUMENTS"


# Puntuadores alternativos (se comparan en la evaluación; el activo se elige por configuración).
SCORERS: dict[str, Callable[[dict], float]] = {
    "cosine": lambda f: f["max_cosine"],
    "coverage": lambda f: f["coverage_best"],
    "blend": lambda f: 0.5 * f["max_cosine"] + 0.5 * f["coverage_best"],
}


@dataclass
class EvidenceAssessment:
    status: EvidenceStatus
    score: float
    features: dict = field(default_factory=dict)
    reason: str = ""
    scorer: str = ""
    thresholds: tuple[float, float] = (0.0, 0.0)


def compute_features(
    query: str, chunks: list[RetrievedChunk], embed: Callable[[list[str]], list[list[float]]] = embed_texts
) -> dict:
    """Calcula las señales de evidencia. `embed` es inyectable para poder probar la lógica."""
    if not chunks:
        return {"max_cosine": 0.0, "top1_cosine": 0.0, "coverage_best": 0.0, "coverage_union": 0.0, "n_chunks": 0}

    terms = set(tokenize(query))
    chunk_tokens = [set(tokenize(c.text)) for c in chunks]
    if terms:
        coverage_best = max(len(terms & t) / len(terms) for t in chunk_tokens)
        coverage_union = len(terms & set().union(*chunk_tokens)) / len(terms)
    else:
        coverage_best = coverage_union = 0.0

    vectores = embed([query] + [c.text for c in chunks])
    q, resto = vectores[0], vectores[1:]
    cosenos = [sum(a * b for a, b in zip(q, v)) for v in resto]  # embeddings normalizados -> coseno
    return {
        "max_cosine": round(max(cosenos), 4),
        "top1_cosine": round(cosenos[0], 4),
        "coverage_best": round(coverage_best, 4),
        "coverage_union": round(coverage_union, 4),
        "n_chunks": len(chunks),
    }


def classify(score: float, low: float, high: float) -> EvidenceStatus:
    if score >= high:
        return EvidenceStatus.SUFFICIENT
    if score >= low:
        return EvidenceStatus.PARTIAL
    return EvidenceStatus.INSUFFICIENT


def assess_evidence(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    store_is_empty: bool = False,
    scorer: str | None = None,
    low: float | None = None,
    high: float | None = None,
    embed: Callable[[list[str]], list[list[float]]] = embed_texts,
) -> EvidenceAssessment:
    scorer = scorer or settings.EVIDENCE_SCORER
    low = settings.EVIDENCE_LOW if low is None else low
    high = settings.EVIDENCE_HIGH if high is None else high

    if store_is_empty:
        return EvidenceAssessment(EvidenceStatus.NO_DOCUMENTS, 0.0, {}, "La base de conocimiento está vacía.", scorer, (low, high))
    if not chunks:
        return EvidenceAssessment(EvidenceStatus.INSUFFICIENT, 0.0, compute_features(query, [], embed), "No se recuperó ningún fragmento.", scorer, (low, high))

    features = compute_features(query, chunks, embed)
    score = round(SCORERS[scorer](features), 4)
    status = classify(score, low, high)
    reason = {
        EvidenceStatus.SUFFICIENT: f"puntaje {score} >= {high}",
        EvidenceStatus.PARTIAL: f"puntaje {score} entre {low} y {high}",
        EvidenceStatus.INSUFFICIENT: f"puntaje {score} < {low}",
    }[status]
    return EvidenceAssessment(status, score, features, f"[{scorer}] {reason}", scorer, (low, high))
