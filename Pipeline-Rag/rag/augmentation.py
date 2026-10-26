"""
Augmentation (Unidad 1): construye el prompt aumentado que se envía al LLM,
siguiendo exactamente la estructura de tres partes vista en el material
del curso: INSTRUCCIONES / CONTEXTO RECUPERADO / PREGUNTA DEL USUARIO.

El LLM nunca accede a la base vectorial directamente: solo recibe el
texto que el Retriever seleccionó.
"""
from rag.vector_store import RetrievedChunk

SYSTEM_INSTRUCTIONS = (
    "Responde la pregunta usando ÚNICAMENTE la información del CONTEXTO "
    "recuperado a continuación. Si la respuesta no se encuentra en el "
    "contexto, dilo explícitamente: no inventes información. "
    "Cuando sea posible, menciona de qué fragmento (archivo y página, si "
    "aplica) proviene cada afirmación."
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Numera y etiqueta cada fragmento recuperado con su fuente, para
    permitir respuestas con citas verificables."""
    partes = []
    for i, chunk in enumerate(chunks, start=1):
        ubicacion = chunk.source + (f", página {chunk.page}" if chunk.page else "")
        partes.append(f"[Fragmento {i} — {ubicacion} | relevancia={chunk.score:.2f}]\n{chunk.text}")
    return "\n\n".join(partes)


SYSTEM_INSTRUCTIONS_V2 = (
    "Responde la pregunta usando ÚNICAMENTE la información del CONTEXTO recuperado a continuación. "
    "Si la respuesta no se encuentra en el contexto, dilo explícitamente: no inventes información. "
    "Cita la fuente de cada afirmación con el número del fragmento entre corchetes, por ejemplo [1] "
    "o [2][3]. Usa solo números de fragmento que aparezcan en el contexto. No cites secciones, "
    "artículos ni páginas por tu cuenta: la ubicación de cada fragmento ya está indicada en el contexto."
)


def format_context_v2(chunks: list[RetrievedChunk]) -> str:
    """Contexto con marcadores [n] que el LLM puede citar y el backend puede validar."""
    partes = []
    for i, chunk in enumerate(chunks, start=1):
        ubicacion = chunk.source + (f", página {chunk.page}" if chunk.page else "")
        partes.append(f"[{i}] Fuente: {ubicacion}\n{chunk.text}")
    return "\n\n".join(partes)


PROMPT_VERSIONS = ("v1", "v2")


def build_prompt(query: str, chunks: list[RetrievedChunk], version: str = "v1") -> str:
    """Ensambla el prompt aumentado final. `v1` es el original (por defecto, línea base);
    `v2` agrega citas numeradas [n] verificables por el backend."""
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"Versión de prompt desconocida '{version}'. Válidas: {PROMPT_VERSIONS}")
    if not chunks:
        contexto = "(No se encontró ningún fragmento relevante en la base de conocimiento.)"
    elif version == "v2":
        contexto = format_context_v2(chunks)
    else:
        contexto = format_context(chunks)

    instrucciones = SYSTEM_INSTRUCTIONS_V2 if version == "v2" else SYSTEM_INSTRUCTIONS
    return (
        f"INSTRUCCIONES\n{instrucciones}\n\n"
        f"CONTEXTO RECUPERADO\n{contexto}\n\n"
        f"PREGUNTA DEL USUARIO\n{query}"
    )