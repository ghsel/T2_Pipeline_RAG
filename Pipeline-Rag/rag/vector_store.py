"""
Almacenamiento vectorial local mediante ChromaDB (modo embebido/persistente,
sin necesidad de levantar un servidor aparte).
"""
import hashlib
from dataclasses import dataclass, field

import chromadb

from config.settings import settings
from rag.chunking import Chunk
from rag.cache import ExactCache, make_key, normalize_query
from rag.embeddings import embed_texts
from rag.spaces import collection_name_for


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int | None
    score: float  # puntaje final del método que lo devolvió, 0 a 1 (más alto = más relevante)
    # --- Trazabilidad (opcionales: los constructores previos siguen siendo válidos) ---
    chunk_id: str | None = None
    doc_type: str | None = None
    method: str = "semantic"  # "semantic" | "bm25" | "hybrid"
    # Puntajes parciales por señal. None = el chunk no estaba entre los candidatos de esa señal.
    scores: dict[str, float | None] = field(default_factory=dict)


class VectorStore:
    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        space: str | None = None,
    ):
        """`space` selecciona un espacio de investigación (colección aislada); si se da
        `collection_name` explícito, ese nombre tiene prioridad."""
        self.client = chromadb.PersistentClient(path=persist_directory or settings.VECTOR_DB_DIR)
        self.space = space
        self.persist_directory = persist_directory or settings.VECTOR_DB_DIR
        self.cache: ExactCache | None = None  # opcional (P1-6); None = comportamiento original
        self.collection = self.client.get_or_create_collection(
            name=collection_name or collection_name_for(space),
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Genera embeddings para los chunks dados y los inserta (upsert) en la colección.
        Con IDs deterministas, insertar dos veces el mismo chunk no lo duplica."""
        if not chunks:
            return
        embeddings = embed_texts([c.text for c in chunks])
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {"source": c.source, "page": c.page or -1, "doc_type": c.doc_type}
                for c in chunks
            ],
        )

    # --- Gestión por documento ------------------------------------------------

    def _source_ids(self, source: str) -> list[str]:
        return self.collection.get(where={"source": source}, include=[])["ids"]

    def delete_source(self, source: str) -> int:
        """Elimina todos los chunks de una fuente. Devuelve cuántos se borraron."""
        ids = self._source_ids(source)
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def replace_source(self, source: str, chunks: list[Chunk]) -> dict:
        """Deja la fuente exactamente con los `chunks` dados: borra los chunks que ya no
        existen y agrega los nuevos. Si el contenido no cambió, no recalcula embeddings."""
        existentes = set(self._source_ids(source))
        nuevos = {c.chunk_id for c in chunks}
        if existentes == nuevos:
            return {"source": source, "deleted": 0, "added": 0, "unchanged": True}

        obsoletos = list(existentes - nuevos)
        if obsoletos:
            self.collection.delete(ids=obsoletos)
        a_agregar = [c for c in chunks if c.chunk_id not in existentes]
        self.add_chunks(a_agregar)
        return {"source": source, "deleted": len(obsoletos), "added": len(a_agregar), "unchanged": False}

    def list_sources(self) -> list[dict]:
        """Resumen por documento: nombre, tipo y número de chunks (ordenado por nombre)."""
        if self.count() == 0:
            return []
        metadatas = self.collection.get(include=["metadatas"])["metadatas"]
        resumen: dict[str, dict] = {}
        for m in metadatas:
            fuente = m.get("source", "desconocido")
            item = resumen.setdefault(
                fuente, {"source": fuente, "doc_type": m.get("doc_type", "unknown"), "n_chunks": 0}
            )
            item["n_chunks"] += 1
        return sorted(resumen.values(), key=lambda r: r["source"])

    def index_version(self) -> str:
        """Huella del contenido indexado (cambia si se agrega, borra o reemplaza algún chunk).
        Sirve para invalidar índices derivados (BM25) y, más adelante, cachés."""
        ids = sorted(self.collection.get(include=[])["ids"])
        return hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest()[:16]

    # --- Consulta ---------------------------------------------------------------

    def _embed_query(self, query_text: str) -> list[float]:
        if self.cache is None:
            return embed_texts([query_text])[0]
        clave = make_key(settings.EMBEDDING_MODEL, normalize_query(query_text))
        vector = self.cache.get("embedding", clave)
        if vector is None:
            vector = embed_texts([query_text])[0]
            self.cache.put("embedding", clave, vector)
        return vector

    def query(self, query_text: str, top_k: int = 4) -> list[RetrievedChunk]:
        """Búsqueda semántica: devuelve los `top_k` chunks más similares a la consulta."""
        if self.count() == 0:
            return []
        query_embedding = self._embed_query(query_text)
        resultados = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

        chunks: list[RetrievedChunk] = []
        for chunk_id, texto, metadata, distancia in zip(
            resultados["ids"][0],
            resultados["documents"][0],
            resultados["metadatas"][0],
            resultados["distances"][0],
        ):
            similitud = max(0.0, 1.0 - distancia)  # distancia coseno -> similitud aprox.
            pagina = metadata.get("page")
            chunks.append(
                RetrievedChunk(
                    text=texto,
                    source=metadata.get("source", "desconocido"),
                    page=None if pagina in (None, -1) else pagina,
                    score=similitud,
                    chunk_id=chunk_id,
                    doc_type=metadata.get("doc_type"),
                    method="semantic",
                    scores={"semantic": similitud},
                )
            )
        return chunks

    def get_all(self) -> list[dict]:
        """Devuelve todos los chunks almacenados (usado para construir el índice BM25)."""
        if self.count() == 0:
            return []
        resultado = self.collection.get(include=["documents", "metadatas"])
        registros = []
        for chunk_id, texto, metadata in zip(
            resultado["ids"], resultado["documents"], resultado["metadatas"]
        ):
            pagina = metadata.get("page")
            registros.append(
                {
                    "chunk_id": chunk_id,
                    "text": texto,
                    "source": metadata.get("source", "desconocido"),
                    "page": None if pagina in (None, -1) else pagina,
                    "doc_type": metadata.get("doc_type"),
                }
            )
        return registros

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Elimina todos los datos de la colección (útil para re-ingestar desde cero)."""
        nombre = self.collection.name
        self.client.delete_collection(nombre)
        self.collection = self.client.get_or_create_collection(
            name=nombre, metadata={"hnsw:space": "cosine"}
        )
