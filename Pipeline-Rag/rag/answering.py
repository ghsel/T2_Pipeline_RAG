"""
Orquestación de una respuesta (P1-1 + P1-2): recuperar -> evaluar evidencia -> (generar) -> validar citas.

Es la única parte que decide el ESTADO de la respuesta, con reglas verificables del backend:

  sin_documentos  la base está vacía.
  sin_evidencia   la política de evidencia rechazó la consulta (no se llamó al LLM) o el propio modelo
                  declaró que el contexto no contiene la respuesta.
  parcial         evidencia débil, o la validación de citas encontró errores / faltan citas (prompt v2).
  respaldada      evidencia suficiente y citas válidas.
  error           falló la generación (Ollama no disponible, etc.).

El generador es inyectable (`generate`), de modo que la lógica se prueba sin Ollama.
"""
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from config.settings import settings
from rag.augmentation import build_prompt
from rag.cache import ExactCache, make_key, normalize_query
from rag.citations import CitationReport, validate_citations
from rag.evidence import EvidenceAssessment, EvidenceStatus, assess_evidence
from rag.generator import OllamaUnavailableError, generate_answer
from rag.retriever import Retriever
from rag.text import strip_accents
from rag.vector_store import RetrievedChunk

_DECLINE_RE = re.compile(
    r"no se (?:menciona|encuentra|indica|define|especifica|proporciona|incluye|dispone|cuenta)|"
    r"no (?:define|especifica|menciona|indica|incluye|hay (?:informacion|datos)|existe informacion|cuenta con|dispongo|puedo (?:responder|determinar))|"
    r"(?:sin|falta de|insuficiente) (?:evidencia|informacion)|lo siento|no tengo (?:informacion|datos)|ningun fragmento"
)


def declares_no_evidence(answer: str) -> bool:
    """Heurística: el modelo admite que el contexto no contiene lo preguntado."""
    return bool(_DECLINE_RE.search(re.sub(r"\s+", " ", strip_accents(answer.lower()))))


@dataclass
class AnswerResult:
    query: str
    status: str
    answer: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    evidence: EvidenceAssessment | None = None
    citations: CitationReport | None = None
    prompt_version: str = "v1"
    policy_enabled: bool = True
    rounds: list[dict] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    error: str | None = None
    llm_called: bool = False
    raw_answer: str = ""  # salida del LLM sin modificar (auditoría); `answer` es lo que se muestra
    cache_hits: list[str] = field(default_factory=list)  # capas de caché que respondieron: 'retrieval' | 'answer'


def _insufficient_message(chunks: list[RetrievedChunk]) -> str:
    texto = ("No encontré evidencia suficiente en los documentos para responder esta pregunta. "
             "Reformúlala o verifica que el documento correspondiente esté cargado.")
    cercanas: list[str] = []
    for c in chunks:
        etiqueta = c.source + (f" (p. {c.page})" if c.page else "")
        if etiqueta not in cercanas:
            cercanas.append(etiqueta)
    if cercanas:
        texto += "\n\nFuentes más cercanas (no respaldan una respuesta): " + ", ".join(cercanas[:3]) + "."
    return texto


def answer_question(
    query: str,
    retriever: Retriever,
    *,
    top_k: int | None = None,
    agentic: bool | None = None,
    use_policy: bool | None = None,
    prompt_version: str | None = None,
    generate: Callable[[str], str] = generate_answer,
    retrieve: Callable[[str, int], list[RetrievedChunk]] | None = None,
    cache: ExactCache | None = None,
) -> AnswerResult:
    top_k = top_k or settings.TOP_K
    use_policy = settings.EVIDENCE_POLICY if use_policy is None else use_policy
    prompt_version = prompt_version or settings.PROMPT_VERSION
    agentic = settings.AGENTIC_MODE if agentic is None else agentic
    t0 = time.perf_counter()
    rounds: list[dict] = []
    cache_hits: list[str] = []

    # Claves de caché: solo con `cache` y la recuperación estándar del Retriever (con un
    # `retrieve` inyectado se desconoce su configuración y no se cachea).
    store_empty = retriever.store.count() == 0
    k_ret = k_ans = None
    if cache is not None and retrieve is None and not store_empty:
        k_ret = make_key(
            "retrieval", retriever.store.collection.name, retriever.store.index_version(),
            normalize_query(query), top_k, agentic, settings.HYBRID_SEARCH, settings.VECTOR_WEIGHT,
            settings.BM25_WEIGHT, settings.MIN_RELEVANCE_SCORE, settings.MAX_RETRIEVAL_ROUNDS,
            settings.EMBEDDING_MODEL, getattr(retriever._tokenize, "__qualname__", "?"),
            settings.GRAPH_ENABLED if retriever.use_graph is None else retriever.use_graph,
            settings.GRAPH_EXTRA, settings.GRAPH_DEPTH, settings.GRAPH_MAX_NODES, settings.GRAPH_NORMALIZE,
            settings.RERANK_ENABLED if retriever.use_rerank is None else retriever.use_rerank,
            settings.RERANK_MODEL, settings.RERANK_CANDIDATES,
        )
        k_ans = make_key(
            "answer", k_ret, settings.OLLAMA_MODEL, settings.GENERATION_TEMPERATURE, prompt_version,
            use_policy, settings.EVIDENCE_SCORER, settings.EVIDENCE_LOW, settings.EVIDENCE_HIGH,
        )
        previa = cache.get("answer", k_ans)
        if previa is not None:
            previa.cache_hits = ["answer"]
            previa.llm_called = False
            previa.timings = {"retrieval_s": 0.0, "evidence_s": 0.0, "generation_s": 0.0,
                              "total_s": round(time.perf_counter() - t0, 4)}
            return previa

    # 1) Recuperación
    if store_empty:
        chunks: list[RetrievedChunk] = []
    elif retrieve is not None:
        chunks = retrieve(query, top_k)
    else:
        guardado = cache.get("retrieval", k_ret) if k_ret else None
        if guardado is not None:
            chunks, rounds = guardado
            cache_hits.append("retrieval")
        elif agentic:
            res = retriever.agentic_search(query, top_k=top_k)
            chunks, rounds = res.chunks, res.rounds
        else:
            chunks = retriever.search(query, top_k=top_k)
        if k_ret and guardado is None:
            cache.put("retrieval", k_ret, (chunks, rounds))
    t_ret = time.perf_counter()

    # 2) Evidencia (siempre se calcula; solo bloquea al LLM si la política está activa)
    evidence = assess_evidence(query, chunks, store_is_empty=store_empty)
    t_ev = time.perf_counter()
    base = dict(query=query, chunks=chunks, evidence=evidence, prompt_version=prompt_version,
                policy_enabled=use_policy, rounds=rounds, cache_hits=cache_hits)

    def _timings(t_end: float, t_gen_start: float | None = None) -> dict:
        return {"retrieval_s": round(t_ret - t0, 4), "evidence_s": round(t_ev - t_ret, 4),
                "generation_s": round(t_end - t_gen_start, 3) if t_gen_start else 0.0,
                "total_s": round(t_end - t0, 3)}

    if evidence.status is EvidenceStatus.NO_DOCUMENTS:
        return AnswerResult(status="sin_documentos", answer="La base de conocimiento está vacía. Incorpora documentos primero.",
                            timings=_timings(time.perf_counter()), **base)
    if use_policy and evidence.status is EvidenceStatus.INSUFFICIENT:
        return AnswerResult(status="sin_evidencia", answer=_insufficient_message(chunks),
                            timings=_timings(time.perf_counter()), **base)

    # 3) Generación
    prompt = build_prompt(query, chunks, version=prompt_version)
    t_g = time.perf_counter()
    try:
        raw = generate(prompt)
    except OllamaUnavailableError as exc:
        return AnswerResult(status="error", answer="", error=str(exc), llm_called=True,
                            timings=_timings(time.perf_counter(), t_g), **base)
    t_end = time.perf_counter()

    # 4) Validación de citas y estado final
    report = validate_citations(raw, chunks, query=query)
    shown = report.text_clean if prompt_version == "v2" else raw
    if declares_no_evidence(raw):
        status = "sin_evidencia"
    elif evidence.status is EvidenceStatus.PARTIAL or report.has_errors:
        status = "parcial"
    elif prompt_version == "v2" and not report.has_valid_citation:
        status = "parcial"
    else:
        status = "respaldada"
    resultado = AnswerResult(status=status, answer=shown, citations=report, llm_called=True, raw_answer=raw,
                             timings=_timings(t_end, t_g), **base)
    if k_ans:
        cache.put("answer", k_ans, resultado)
    return resultado
