"""
Validación de citas (P1-2): comprueba, con reglas verificables, que las referencias de la respuesta
del LLM correspondan a fragmentos que REALMENTE se recuperaron.

El LLM puede citar `[1]`, `[2][3]`, `[1, 3]` o "Fragmento 2". Aquí se resuelve cada número contra la
lista de chunks recuperados (1 = primero). Una cita fuera de rango es inválida y se elimina del texto
mostrado. Además se señalan referencias que el contexto no respalda (secciones, archivos, números).

Límites: es una validación formal y léxica. Que el fragmento citado contenga la cifra no prueba que la
afirmación completa sea correcta; por eso los problemas no críticos se reportan como `warning`.
"""
import re
from dataclasses import dataclass, field

from rag.text import strip_accents
from rag.vector_store import RetrievedChunk

_BRACKET_RE = re.compile(r"\[\s*(?:fragmentos?\s*)?(\d+(?:\s*(?:,|;|y|-|–)\s*\d+)*)\s*\]", re.IGNORECASE)
_FREE_FRAGMENT_RE = re.compile(r"\bfragmentos?\s+(\d+)", re.IGNORECASE)
_SECTION_RE = re.compile(r"\b(seccion|articulo|capitulo|numeral)\s+(\d+(?:\.\d+)*)")
_FILE_RE = re.compile(r"\b[\w\-]+\.(?:md|html|pdf)\b")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


@dataclass
class CitationReport:
    citations: list[dict] = field(default_factory=list)   # {marker, n, valid, chunk_id, source, page}
    text_clean: str = ""                                   # respuesta sin marcadores inválidos
    issues: list[dict] = field(default_factory=list)       # {type, severity: error|warning, detail}

    @property
    def cited_chunk_ids(self) -> list[str]:
        vistos: list[str] = []
        for c in self.citations:
            if c["valid"] and c["chunk_id"] and c["chunk_id"] not in vistos:
                vistos.append(c["chunk_id"])
        return vistos

    @property
    def has_valid_citation(self) -> bool:
        return any(c["valid"] for c in self.citations)

    @property
    def has_errors(self) -> bool:
        return any(i["severity"] == "error" for i in self.issues)


def _expand(group: str) -> list[int]:
    numeros: list[int] = []
    for parte in re.split(r"\s*(?:,|;|\sy\s)\s*", group.strip()):
        rango = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", parte)
        if rango:
            a, b = int(rango.group(1)), int(rango.group(2))
            numeros.extend(range(a, min(b, a + 9) + 1))
        elif parte.strip().isdigit():
            numeros.append(int(parte))
    return numeros


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(text.lower())).strip()


def validate_citations(answer: str, chunks: list[RetrievedChunk], query: str = "") -> CitationReport:
    """Valida las citas de `answer` contra los `chunks` recuperados (en el orden en que se mostraron)."""
    report = CitationReport(text_clean=answer)
    n_chunks = len(chunks)

    def _registrar(marker: str, n: int) -> bool:
        valido = 1 <= n <= n_chunks
        chunk = chunks[n - 1] if valido else None
        report.citations.append({
            "marker": marker, "n": n, "valid": valido,
            "chunk_id": chunk.chunk_id if chunk else None,
            "source": chunk.source if chunk else None, "page": chunk.page if chunk else None,
        })
        if not valido:
            report.issues.append({"type": "invalid_citation", "severity": "error",
                                  "detail": f"{marker} no corresponde a ningún fragmento recuperado (hay {n_chunks})"})
        return valido

    # Citas entre corchetes: se eliminan del texto mostrado las que no existen.
    def _reemplazo(m: re.Match) -> str:
        numeros = _expand(m.group(1))
        validos = [_registrar(m.group(0), n) for n in numeros]
        return m.group(0) if all(validos) else ""

    report.text_clean = _BRACKET_RE.sub(_reemplazo, answer)

    # Menciones libres "Fragmento N" (típicas del prompt original).
    sin_corchetes = _BRACKET_RE.sub(" ", answer)
    for m in _FREE_FRAGMENT_RE.finditer(sin_corchetes):
        _registrar(m.group(0), int(m.group(1)))

    contexto = _norm("\n".join(c.text for c in chunks))

    # Referencias estructurales que el contexto no respalda.
    texto_norm = _norm(sin_corchetes)
    for tipo, num in _SECTION_RE.findall(texto_norm):
        respaldada = re.search(rf"(?:^|\s){re.escape(num)}\.\s", contexto) or f"{tipo} {num}" in contexto
        if not respaldada:
            report.issues.append({"type": "fabricated_reference", "severity": "error",
                                  "detail": f"«{tipo} {num}» no aparece en los fragmentos recuperados"})
    fuentes = {c.source.lower() for c in chunks}
    for archivo in sorted(set(_FILE_RE.findall(answer.lower()))):
        if archivo not in fuentes:
            report.issues.append({"type": "fabricated_reference", "severity": "error",
                                  "detail": f"el archivo «{archivo}» no está entre las fuentes recuperadas"})

    # Cifras de la respuesta que no están en los fragmentos citados (o en todos, si no hay citas).
    citados = [chunks[c["n"] - 1] for c in report.citations if c["valid"]]
    base = _norm("\n".join(c.text for c in (citados or chunks)) + " " + query)
    base_nums = set(_NUM_RE.findall(base))
    sin_respaldo = sorted({n for n in _NUM_RE.findall(texto_norm) if len(n) > 1 and n not in base_nums})
    for n in sin_respaldo:
        report.issues.append({"type": "unsupported_number", "severity": "warning",
                              "detail": f"la cifra «{n}» no aparece en los fragmentos {'citados' if citados else 'recuperados'}"})

    if not report.citations:
        report.issues.append({"type": "no_citations", "severity": "warning", "detail": "la respuesta no cita ningún fragmento"})
    return report
