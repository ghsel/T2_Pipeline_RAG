"""
Expansión por relaciones (P2-2, experimental y opcional). NO es un Graph RAG completo.

Construye con NetworkX un grafo de ENTIDADES a partir de los chunks indexados:

  - Entidades por reglas (sin NER entrenado ni LLM): productos, instancias de aprobación,
    conceptos del dominio bancario y valores literales (porcentajes y montos).
  - Relación entre dos entidades = co-ocurrencia en un mismo chunk; cada arista conserva los
    chunks que la justifican (evidencia textual).

En consulta: parte de las entidades de la pregunta y de los primeros fragmentos recuperados,
recorre el grafo hasta `depth` saltos (acotado a `max_nodes` entidades, ignorando entidades
«hub» presentes en demasiados chunks) y propone hasta `extra` fragmentos que la recuperación
base no devolvió. Esos fragmentos REEMPLAZAN los últimos lugares del top-k: el tamaño del
contexto es el mismo que sin grafo (comparación justa).

El grafo se persiste en JSON versionado por `VectorStore.index_version()`; si el corpus cambia,
se reconstruye.
"""
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from rag.text import strip_accents

# (nombre canónico, tipo, patrón sobre texto en minúsculas y sin tildes)
_RULES: list[tuple[str, str, str]] = [
    ("CVIS (vivienda de interés social)", "producto", r"vivienda de interes social|\bcvis\b"),
    ("Crédito de consumo", "producto", r"credito(?:s)? de consumo"),
    ("Microcrédito", "producto", r"microcredito"),
    ("Crédito PYME", "producto", r"credito(?:s)? pyme|\bpyme\b"),
    ("Caja de ahorro", "producto", r"caja(?:s)? de ahorro"),
    ("Cuenta Ahorro Futuro", "producto", r"ahorro futuro"),
    ("Cuenta corriente", "producto", r"cuenta corriente"),
    ("Depósito a plazo fijo", "producto", r"deposito(?:s)? a plazo fijo|\bdpf\b"),
    ("Tarjeta de débito", "producto", r"tarjeta(?:s)? de debito"),
    ("Banca móvil", "producto", r"banca movil|aplicacion movil"),
    ("Transferencias", "producto", r"transferencia"),
    ("Jefe de Agencia", "instancia", r"jefe de agencia"),
    ("Gerente Regional", "instancia", r"gerente regional"),
    ("Comité Nacional de Créditos", "instancia", r"comite nacional"),
    ("Gerencia de Cumplimiento", "instancia", r"gerencia de cumplimiento|area de cumplimiento"),
    ("ASFI", "instancia", r"\basfi\b|autoridad de supervision"),
    ("Mora", "concepto", r"\bmora\b|atraso|cartera vencida"),
    ("Reprogramación", "concepto", r"reprogramacion|reprogramar|reprogramado"),
    ("Refinanciamiento", "concepto", r"refinanci"),
    ("Aprobación", "concepto", r"aprobacion|aprobad[oa]s?|aprobo"),
    ("Comisión de apertura", "concepto", r"comision de apertura"),
    ("Prepago", "concepto", r"prepago|pague? por adelantado"),
    ("Seguro de desgravamen", "concepto", r"desgravamen"),
    ("Tarifario", "concepto", r"tarifario"),
    ("Comunicado de actualización", "concepto", r"comunicado"),
    ("Apertura de cuentas / KYC", "concepto", r"apertura de cuenta|conocimiento del cliente|abrir una cuenta|abrir cuenta"),
    ("Garantía", "concepto", r"garantia"),
    ("Cobranza", "concepto", r"cobranza"),
    ("Tasa de interés", "concepto", r"\btasa"),
    ("Menores de edad", "concepto", r"menor(?:es)? de edad|menores|ninos|\bhijo\b"),
]
_COMPILED = [(n, t, re.compile(p)) for n, t, p in _RULES]
_PCT = re.compile(r"\d+(?:,\d+)?\s?%")
_AMOUNT = re.compile(r"\b(bs|usd|ufv)\.?\s?(\d[\d\.]*\d|\d)")


def extract_entities(text: str) -> dict[str, str]:
    """{entidad: tipo} encontradas en un texto, por reglas deterministas."""
    t = strip_accents(text.lower())
    found = {n: tipo for n, tipo, rx in _COMPILED if rx.search(t)}
    for m in _PCT.finditer(t):
        found["% " + m.group().replace(" ", "")[:-1]] = "valor"
    for m in _AMOUNT.finditer(t):
        found[f"{m.group(1).upper()} {m.group(2)}"] = "valor"
    return found


@dataclass
class GraphTrace:
    """Explicación de una expansión (para CLI/UI y para el informe)."""
    query_entities: list[str] = field(default_factory=list)
    seed_entities: list[str] = field(default_factory=list)
    reached: dict[str, int] = field(default_factory=dict)  # entidad -> saltos desde la semilla
    added: list[dict] = field(default_factory=list)        # {chunk_id, source, score, via}
    note: str = ""


class GraphIndex:
    def __init__(self, records: list[dict], version: str = ""):
        self.version = version
        self.records = {r["chunk_id"]: r for r in records}
        self.chunk_entities: dict[str, dict[str, str]] = {
            r["chunk_id"]: extract_entities(r["text"]) for r in records
        }
        self.entity_chunks: dict[str, set[str]] = {}
        self.entity_type: dict[str, str] = {}
        for cid, ents in self.chunk_entities.items():
            for e, tipo in ents.items():
                self.entity_chunks.setdefault(e, set()).add(cid)
                self.entity_type[e] = tipo
        self.graph = nx.Graph()
        self.graph.add_nodes_from((e, {"type": t}) for e, t in self.entity_type.items())
        for cid, ents in self.chunk_entities.items():
            nombres = sorted(ents)
            for i, a in enumerate(nombres):
                for b in nombres[i + 1:]:
                    if self.graph.has_edge(a, b):
                        self.graph[a][b]["weight"] += 1
                        if len(self.graph[a][b]["evidence"]) < 5:
                            self.graph[a][b]["evidence"].append(cid)
                    else:
                        self.graph.add_edge(a, b, weight=1, evidence=[cid])

    # --- Persistencia ------------------------------------------------------------

    def to_json(self) -> dict:
        return {"version": self.version, "records": list(self.records.values())}

    @classmethod
    def load_or_build(cls, records: list[dict], version: str, cache_dir: Path | None, name: str = "graph") -> "GraphIndex":
        """Reutiliza el JSON persistido si su versión coincide con la del índice; si no, reconstruye."""
        ruta = Path(cache_dir) / f"{name}.json" if cache_dir else None
        if ruta and ruta.exists():
            try:
                data = json.loads(ruta.read_text(encoding="utf-8"))
                if data.get("version") == version:
                    return cls(data["records"], version)
            except (OSError, ValueError, KeyError):
                pass
        g = cls(records, version)
        if ruta:
            ruta.parent.mkdir(parents=True, exist_ok=True)
            ruta.write_text(json.dumps(g.to_json(), ensure_ascii=False), encoding="utf-8")
        return g

    # --- Expansión -------------------------------------------------------------------

    def stats(self) -> dict:
        tipos: dict[str, int] = {}
        for t in self.entity_type.values():
            tipos[t] = tipos.get(t, 0) + 1
        top = sorted(self.entity_chunks.items(), key=lambda kv: -len(kv[1]))[:5]
        return {"chunks": len(self.records), "entities": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(), "by_type": tipos,
                "most_frequent": [(e, len(c)) for e, c in top]}

    def expand(self, query: str, seed_chunk_ids: list[str], extra: int = 2, depth: int = 2,
               max_nodes: int = 12, seed_top: int = 2, hub_ratio: float = 0.35,
               normalize: bool = False) -> tuple[list[tuple[str, float]], GraphTrace]:
        """Devuelve [(chunk_id, puntaje)] de hasta `extra` fragmentos NO presentes en `seed_chunk_ids`.
        `normalize` divide el puntaje por la raíz del nº de entidades del chunk (evita que los chunks
        con muchas entidades dominen siempre, igual que la normalización por longitud en BM25)."""
        n = max(len(self.records), 1)
        trace = GraphTrace()
        es_hub = lambda e: len(self.entity_chunks.get(e, ())) / n > hub_ratio  # noqa: E731

        q_ents = [e for e in extract_entities(query) if e in self.entity_chunks]
        trace.query_entities = q_ents
        pesos: dict[str, float] = {e: 1.0 for e in q_ents if not es_hub(e)}
        for cid in seed_chunk_ids[:seed_top]:
            for e in self.chunk_entities.get(cid, {}):
                if not es_hub(e):
                    pesos[e] = max(pesos.get(e, 0.0), 0.5)
        trace.seed_entities = sorted(pesos)
        if not pesos:
            trace.note = "Sin entidades semilla (o todas son hubs): no se expande."
            return [], trace

        # Recorrido por saltos desde las semillas, sin atravesar hubs.
        visibles = self.graph.subgraph(e for e in self.graph.nodes if not es_hub(e))
        alcanzadas: dict[str, tuple[float, int]] = {e: (w, 0) for e, w in pesos.items()}
        for semilla, w in pesos.items():
            for e, d in nx.single_source_shortest_path_length(visibles, semilla, cutoff=depth).items():
                peso = w * (0.5 ** d)
                if e not in alcanzadas or peso > alcanzadas[e][0]:
                    alcanzadas[e] = (peso, d)
        if len(alcanzadas) > max_nodes:
            alcanzadas = dict(sorted(alcanzadas.items(), key=lambda kv: -kv[1][0])[:max_nodes])
        trace.reached = {e: d for e, (_, d) in alcanzadas.items()}

        vistos = set(seed_chunk_ids)
        puntajes: dict[str, float] = {}
        via: dict[str, list[str]] = {}
        for e, (peso, _) in alcanzadas.items():
            idf = math.log(1 + n / len(self.entity_chunks[e]))
            for cid in self.entity_chunks[e]:
                if cid in vistos:
                    continue
                puntajes[cid] = puntajes.get(cid, 0.0) + peso * idf
                via.setdefault(cid, []).append(e)
        if normalize:
            puntajes = {c: s / math.sqrt(len(self.chunk_entities[c])) for c, s in puntajes.items()}
        ranking = sorted(puntajes.items(), key=lambda kv: (-kv[1], kv[0]))[:extra]
        trace.added = [{"chunk_id": c, "source": self.records[c]["source"], "score": round(s, 3), "via": via[c][:4]}
                       for c, s in ranking]
        return ranking, trace
