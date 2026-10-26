"""
Configuración centralizada del proyecto RAG.

Todo corre localmente por defecto: el modelo de embeddings se descarga
una vez desde Hugging Face (open source, gratuito) y la generación usa
Ollama en tu propia máquina (modelos open-weight, sin costo por token).
No se requiere ninguna clave de API para el funcionamiento por defecto.
"""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # --- Rutas ---
    RAW_DATA_DIR: str = os.getenv("RAW_DATA_DIR", "./data/raw")
    PROCESSED_DATA_DIR: str = os.getenv("PROCESSED_DATA_DIR", "./data/processed")
    VECTOR_DB_DIR: str = os.getenv("VECTOR_DB_DIR", "./data/chroma_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "base_conocimiento")

    # --- Chunking (Unidad 3) ---
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))         # caracteres por chunk
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120"))   # solapamiento entre chunks

    # --- Embeddings (Unidad 1) ---
    # Modelo local, multilingüe, liviano (~470MB, se descarga una sola vez).
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )

    # --- Retrieval (Unidad 1) ---
    TOP_K: int = int(os.getenv("TOP_K", "4"))
    HYBRID_SEARCH: bool = os.getenv("HYBRID_SEARCH", "true").lower() == "true"
    VECTOR_WEIGHT: float = float(os.getenv("VECTOR_WEIGHT", "0.6"))  # peso búsqueda semántica
    BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.4"))      # peso búsqueda por palabras clave

    # --- Agentic RAG (Unidad 2 · tendencias 2026) ---
    AGENTIC_MODE: bool = os.getenv("AGENTIC_MODE", "true").lower() == "true"
    MIN_RELEVANCE_SCORE: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.35"))
    MAX_RETRIEVAL_ROUNDS: int = int(os.getenv("MAX_RETRIEVAL_ROUNDS", "2"))

    # --- Política de evidencia y citas (P1-1 / P1-2) ---
    # EVIDENCE_POLICY=false conserva el comportamiento original (siempre llama al LLM).
    EVIDENCE_POLICY: bool = os.getenv("EVIDENCE_POLICY", "true").lower() == "true"
    EVIDENCE_SCORER: str = os.getenv("EVIDENCE_SCORER", "cosine")  # cosine | coverage | blend
    EVIDENCE_LOW: float = float(os.getenv("EVIDENCE_LOW", "0.35"))    # por debajo: INSUFFICIENT (percentil 5 de respondibles de calibración)
    EVIDENCE_HIGH: float = float(os.getenv("EVIDENCE_HIGH", "0.60"))  # desde aquí: SUFFICIENT (percentil 20)
    PROMPT_VERSION: str = os.getenv("PROMPT_VERSION", "v2")  # v1 = original, v2 = con citas [n]

    # --- Expansión por relaciones / grafo (P2-2, experimental; desactivada por defecto) ---
    GRAPH_ENABLED: bool = os.getenv("GRAPH_ENABLED", "false").lower() == "true"
    GRAPH_EXTRA: int = int(os.getenv("GRAPH_EXTRA", "2"))        # fragmentos que el grafo puede aportar al top-k
    GRAPH_DEPTH: int = int(os.getenv("GRAPH_DEPTH", "2"))        # saltos máximos entre entidades
    GRAPH_MAX_NODES: int = int(os.getenv("GRAPH_MAX_NODES", "12"))
    GRAPH_NORMALIZE: bool = os.getenv("GRAPH_NORMALIZE", "false").lower() == "true"  # puntaje de chunk / raíz(nº entidades)

    # --- Reranking con cross-encoder local (P2-1, experimental; desactivado por defecto) ---
    RERANK_ENABLED: bool = os.getenv("RERANK_ENABLED", "false").lower() == "true"
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    RERANK_CANDIDATES: int = int(os.getenv("RERANK_CANDIDATES", "8"))  # candidatos que ve el reranker

    # --- Generación (Ollama, local y gratuito) ---
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    GENERATION_TEMPERATURE: float = float(os.getenv("GENERATION_TEMPERATURE", "0.2"))


settings = Settings()
