"""
El Retriever (Unidad 1): encuentra los fragmentos más relevantes para una
consulta dada.

Implementa dos mejoras señaladas como tendencias 2026 en el material:

  - Búsqueda híbrida: combina similitud vectorial (densa) con BM25
    (coincidencia de palabras clave), ponderando ambos puntajes.
  - Agentic RAG (versión simplificada): en vez de recuperar una sola vez,
    evalúa si la evidencia encontrada es suficientemente relevante y, si
    no lo es, reintenta ampliando la búsqueda — sin necesidad de un
    framework de agentes externo (LangGraph, CrewAI, etc. se mencionan en
    el material como alternativas más completas para flujos complejos).
"""
from collections.abc import Callable
from dataclasses import dataclass

from pathlib import Path

from rank_bm25 import BM25Okapi

from config.settings import settings
from rag.graph import GraphIndex, GraphTrace
from rag.reranker import rerank
from rag.text import tokenize
from rag.vector_store import RetrievedChunk, VectorStore


@dataclass
class AgenticSearchResult:
    chunks: list[RetrievedChunk]
    rounds: list[dict]  # traza de cada ronda de recuperación (para transparencia/debug)


class Retriever:
    def __init__(self, vector_store: VectorStore, tokenizer: Callable[[str], list[str]] | None = None,
                 use_graph: bool | None = None, use_rerank: bool | None = None):
        self.store = vector_store
        # Inyectable para poder reproducir la tokenización original en experimentos.
        self._tokenize = tokenizer or tokenize
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[dict] = []
        self._bm25_version: str | None = None
        self.use_graph = use_graph  # None -> settings.GRAPH_ENABLED
        self.use_rerank = use_rerank  # None -> settings.RERANK_ENABLED (tiene prioridad sobre el grafo)
        self._graph: GraphIndex | None = None
        self.last_graph_trace: GraphTrace | None = None

    # --- Búsqueda semántica pura -------------------------------------------

    def semantic_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return self.store.query(query, top_k=top_k)

    # --- Búsqueda por palabras clave (BM25) ---------------------------------

    def _ensure_bm25_index(self) -> None:
        """Construye el índice BM25 y lo reconstruye si el contenido de la base cambió
        (altas, bajas o reemplazos de documentos), para no servir resultados obsoletos."""
        version = self.store.index_version()
        if self._bm25 is not None and version == self._bm25_version:
            return
        self._bm25_corpus = self.store.get_all()
        corpus_tokenizado = [self._tokenize(r["text"]) for r in self._bm25_corpus]
        self._bm25 = BM25Okapi(corpus_tokenizado) if corpus_tokenizado else None
        self._bm25_version = version

    def keyword_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self._ensure_bm25_index()
        if self._bm25 is None or not self._bm25_corpus:
            return []

        puntajes = self._bm25.get_scores(self._tokenize(query))
        maximo = max(puntajes) if len(puntajes) and max(puntajes) > 0 else 1.0

        ranking = sorted(
            zip(puntajes, self._bm25_corpus), key=lambda par: par[0], reverse=True
        )[:top_k]

        return [
            RetrievedChunk(
                text=r["text"], source=r["source"], page=r["page"], score=float(s / maximo),
                chunk_id=r.get("chunk_id"), doc_type=r.get("doc_type"), method="bm25",
                scores={"bm25": float(s / maximo), "bm25_raw": float(s)},
            )
            for s, r in ranking
        ]

    # --- Búsqueda híbrida ----------------------------------------------------

    def hybrid_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Suma ponderada de similitud coseno y BM25 normalizado (fórmula original,
        conservada como configuración de referencia). Cada resultado expone sus
        puntajes parciales; None indica que no estaba entre los candidatos de esa señal."""
        semantico = self.semantic_search(query, top_k=top_k * 2)
        por_palabra_clave = self.keyword_search(query, top_k=top_k * 2)

        combinados: dict[str, dict] = {}

        def _entrada(r: RetrievedChunk) -> dict:
            clave = r.chunk_id or f"{r.source}|{r.page}|{r.text}"
            return combinados.setdefault(
                clave, {"chunk": r, "total": 0.0, "semantic": None, "bm25": None}
            )

        for r in semantico:
            e = _entrada(r)
            e["semantic"] = r.score
            e["total"] += settings.VECTOR_WEIGHT * r.score
        for r in por_palabra_clave:
            e = _entrada(r)
            e["bm25"] = r.score
            e["total"] += settings.BM25_WEIGHT * r.score

        ranking = sorted(combinados.values(), key=lambda e: e["total"], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                text=e["chunk"].text, source=e["chunk"].source, page=e["chunk"].page,
                score=e["total"], chunk_id=e["chunk"].chunk_id, doc_type=e["chunk"].doc_type,
                method="hybrid",
                scores={"semantic": e["semantic"], "bm25": e["bm25"], "hybrid": e["total"]},
            )
            for e in ranking
        ]

    def hybrid_rrf_search(
        self, query: str, top_k: int, rrf_k: int = 60, pool_factor: int = 2
    ) -> list[RetrievedChunk]:
        """Fusión por rangos (Reciprocal Rank Fusion): suma 1/(rrf_k + posición) de cada
        lista. No depende de la escala de los puntajes. `score` queda en escala RRF
        (≈0.01–0.03), NO comparable con los umbrales de coseno/BM25 normalizado."""
        listas = {
            "semantic": self.semantic_search(query, top_k=top_k * pool_factor),
            "bm25": self.keyword_search(query, top_k=top_k * pool_factor),
        }
        combinados: dict[str, dict] = {}
        for nombre, lista in listas.items():
            if nombre == "bm25":
                # Puntaje 0 = sin coincidencia léxica: no debe ganar crédito por posición.
                lista = [r for r in lista if r.scores.get("bm25_raw", 0.0) > 0]
            for posicion, r in enumerate(lista, start=1):
                clave = r.chunk_id or f"{r.source}|{r.page}|{r.text}"
                e = combinados.setdefault(
                    clave, {"chunk": r, "total": 0.0, "semantic": None, "bm25": None}
                )
                e[nombre] = r.score
                e["total"] += 1.0 / (rrf_k + posicion)

        ranking = sorted(combinados.values(), key=lambda e: e["total"], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                text=e["chunk"].text, source=e["chunk"].source, page=e["chunk"].page,
                score=e["total"], chunk_id=e["chunk"].chunk_id, doc_type=e["chunk"].doc_type,
                method="hybrid_rrf",
                scores={"semantic": e["semantic"], "bm25": e["bm25"], "rrf": e["total"]},
            )
            for e in ranking
        ]

    # --- Reranking con cross-encoder (experimental) --------------------------------

    def rerank_search(self, query: str, top_k: int, candidates: int | None = None,
                      model: str | None = None) -> list[RetrievedChunk]:
        """Híbrida con `candidates` resultados, reordenados por un cross-encoder local."""
        candidatos = max(candidates or settings.RERANK_CANDIDATES, top_k)
        return rerank(query, self.hybrid_search(query, candidatos), top_k, model or settings.RERANK_MODEL)

    # --- Expansión por relaciones (experimental) ----------------------------------

    def get_graph(self) -> GraphIndex:
        """Grafo de entidades del corpus actual (persistido y versionado por `index_version`)."""
        version = self.store.index_version()
        if self._graph is None or self._graph.version != version:
            carpeta = Path(self.store.persist_directory) / "graph"
            self._graph = GraphIndex.load_or_build(
                self.store.get_all(), version, carpeta, name=self.store.collection.name
            )
        return self._graph

    def graph_search(self, query: str, top_k: int, extra: int | None = None, normalize: bool | None = None) -> list[RetrievedChunk]:
        """Híbrida + expansión por relaciones. Los últimos `extra` lugares del top-k se cambian por
        fragmentos propuestos por el grafo (si los hay): el tamaño del contexto no aumenta.
        `self.last_graph_trace` explica qué entidades y relaciones se usaron."""
        extra = settings.GRAPH_EXTRA if extra is None else extra
        base = self.hybrid_search(query, top_k)
        if self.store.count() == 0 or extra <= 0 or not base:
            self.last_graph_trace = GraphTrace(note="Sin expansión.")
            return base
        grafo = self.get_graph()
        propuestos, traza = grafo.expand(
            query, [c.chunk_id for c in base if c.chunk_id], extra=extra,
            depth=settings.GRAPH_DEPTH, max_nodes=settings.GRAPH_MAX_NODES,
            normalize=settings.GRAPH_NORMALIZE if normalize is None else normalize,
        )
        self.last_graph_trace = traza
        if not propuestos:
            return base
        conservados = base[: max(top_k - len(propuestos), 0)]
        piso = min(c.score for c in conservados) if conservados else 0.0
        nuevos = []
        for cid, puntaje in propuestos:
            r = grafo.records[cid]
            nuevos.append(RetrievedChunk(
                text=r["text"], source=r["source"], page=r["page"], score=piso, chunk_id=cid,
                doc_type=r.get("doc_type"), method="graph_expansion",
                scores={"semantic": None, "bm25": None, "graph": puntaje},
            ))
        return conservados + nuevos

    # --- Punto de entrada estándar --------------------------------------------

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or settings.TOP_K
        usar_rerank = settings.RERANK_ENABLED if self.use_rerank is None else self.use_rerank
        if usar_rerank and settings.HYBRID_SEARCH:
            return self.rerank_search(query, top_k)
        usar_grafo = settings.GRAPH_ENABLED if self.use_graph is None else self.use_graph
        if usar_grafo and settings.HYBRID_SEARCH:
            return self.graph_search(query, top_k)
        if settings.HYBRID_SEARCH:
            return self.hybrid_search(query, top_k)
        return self.semantic_search(query, top_k)

    # --- Agentic RAG (tendencia 2026) -----------------------------------------

    def agentic_search(self, query: str, top_k: int | None = None) -> AgenticSearchResult:
        """
        Bucle simplificado de "Analizar pregunta → ¿Necesito más datos? →
        Recuperar y evaluar → (reformular si la evidencia es débil)".

        La "reformulación" aquí es deliberadamente simple (amplía el número
        de resultados solicitados); en un sistema de producción se usaría
        un LLM o un framework de agentes para reformular semánticamente
        la consulta, tal como señala el material del curso.
        """
        top_k = top_k or settings.TOP_K
        current_k = top_k
        rounds: list[dict] = []
        results: list[RetrievedChunk] = []

        for numero_ronda in range(1, settings.MAX_RETRIEVAL_ROUNDS + 1):
            results = self.search(query, top_k=current_k)
            score_promedio = (
                sum(r.score for r in results) / len(results) if results else 0.0
            )
            rounds.append(
                {
                    "round": numero_ronda,
                    "top_k": current_k,
                    "avg_score": round(score_promedio, 3),
                    "n_results": len(results),
                }
            )

            evidencia_suficiente = score_promedio >= settings.MIN_RELEVANCE_SCORE
            es_ultima_ronda = numero_ronda == settings.MAX_RETRIEVAL_ROUNDS
            if evidencia_suficiente or es_ultima_ronda:
                break

            current_k += 2  # "reformulación": amplía la búsqueda

        return AgenticSearchResult(chunks=results, rounds=rounds)
