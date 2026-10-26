"""
Caché EXACTA en memoria (P1-6), con tres capas independientes:

  embedding  texto de consulta -> vector                (clave: modelo de embeddings + texto)
  retrieval  consulta -> fragmentos recuperados         (clave: espacio, versión del índice, parámetros de recuperación)
  answer     consulta -> respuesta completa             (clave: la de retrieval + modelo, prompt, política y umbrales)

Las claves incluyen la versión del índice (`VectorStore.index_version`): al agregar, borrar o
reemplazar documentos la clave cambia y las entradas viejas dejan de coincidir (no se sirven
resultados obsoletos). Solo hay coincidencia si la consulta es idéntica salvo espacios
sobrantes; NO es una caché semántica. Se descartan las entradas más antiguas (LRU).

Limitación conocida: la generación usa temperatura > 0, así que un acierto de la capa `answer`
repite una respuesta anterior en vez de generar otra. Las respuestas con error no se guardan.
"""
import copy
import hashlib
import json
import re
from collections import OrderedDict

LAYERS = ("embedding", "retrieval", "answer")


def normalize_query(query: str) -> str:
    """Solo quita espacios sobrantes; mayúsculas y signos se conservan (caché exacta)."""
    return re.sub(r"\s+", " ", query).strip()


def make_key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")).hexdigest()


class ExactCache:
    def __init__(self, max_entries: int = 512):
        self.max_entries = max_entries
        self._data: dict[str, OrderedDict] = {layer: OrderedDict() for layer in LAYERS}
        self.hits = dict.fromkeys(LAYERS, 0)
        self.misses = dict.fromkeys(LAYERS, 0)

    def get(self, layer: str, key: str):
        capa = self._data[layer]
        if key in capa:
            capa.move_to_end(key)
            self.hits[layer] += 1
            return copy.deepcopy(capa[key])  # el llamador no puede alterar lo almacenado
        self.misses[layer] += 1
        return None

    def put(self, layer: str, key: str, value) -> None:
        capa = self._data[layer]
        capa[key] = copy.deepcopy(value)
        capa.move_to_end(key)
        while len(capa) > self.max_entries:
            capa.popitem(last=False)

    def clear(self, layer: str | None = None) -> None:
        for nombre in ([layer] if layer else LAYERS):
            self._data[nombre].clear()

    def stats(self) -> dict:
        out = {}
        for layer in LAYERS:
            total = self.hits[layer] + self.misses[layer]
            out[layer] = {"hits": self.hits[layer], "misses": self.misses[layer],
                          "hit_rate": round(self.hits[layer] / total, 3) if total else None,
                          "entries": len(self._data[layer])}
        return out
