"""
Interfaz web de chat (Streamlit, open source) para interactuar con el
sistema RAG, mostrando también los fragmentos recuperados y su fuente
exacta (citas verificables, ver Unidad 1).

Ejecutar con:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from config.settings import settings  # noqa: E402
from rag.augmentation import build_prompt  # noqa: E402
from rag.generator import OllamaUnavailableError, generate_answer  # noqa: E402
from rag.retriever import Retriever  # noqa: E402
from rag.vector_store import VectorStore  # noqa: E402

st.set_page_config(page_title="RAG — Base de Conocimiento", page_icon="🔎", layout="wide")

st.title("🔎 Sistema RAG — Base de Conocimiento")
st.caption(
    "Retrieval-Augmented Generation · 100% open source · embeddings y generación corren "
    "localmente (sin API de pago)"
)


@st.cache_resource
def get_retriever() -> tuple[Retriever, VectorStore]:
    store = VectorStore()
    return Retriever(store), store


retriever, store = get_retriever()

with st.sidebar:
    st.header("Configuración")
    agentic = st.checkbox("Modo Agentic RAG", value=settings.AGENTIC_MODE)
    top_k = st.slider("Fragmentos a recuperar (top-k)", 1, 10, settings.TOP_K)
    st.divider()
    st.metric("Chunks en la base de conocimiento", store.count())
    if store.count() == 0:
        st.warning("La base está vacía. Ejecuta en una terminal: `python cli.py ingest`")
    st.divider()
    st.caption(f"Modelo de embeddings: `{settings.EMBEDDING_MODEL}`")
    st.caption(f"Modelo de generación: `{settings.OLLAMA_MODEL}` (vía Ollama)")

if "history" not in st.session_state:
    st.session_state.history = []

for mensaje in st.session_state.history:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])

pregunta = st.chat_input("Escribe tu pregunta sobre la base de conocimiento...")

if pregunta:
    st.session_state.history.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        if store.count() == 0:
            respuesta = "La base de conocimiento está vacía. Ejecuta primero `python cli.py ingest`."
            st.warning(respuesta)
        else:
            with st.spinner("Recuperando fragmentos relevantes..."):
                if agentic:
                    resultado = retriever.agentic_search(pregunta, top_k=top_k)
                    chunks = resultado.chunks
                else:
                    chunks = retriever.search(pregunta, top_k=top_k)

            with st.expander(f"📄 {len(chunks)} fragmento(s) recuperado(s) — ver fuentes"):
                for i, c in enumerate(chunks, start=1):
                    ubicacion = c.source + (f", página {c.page}" if c.page else "")
                    st.markdown(f"**[{i}] {ubicacion}** · relevancia={c.score:.2f}")
                    texto_corto = c.text[:400] + ("..." if len(c.text) > 400 else "")
                    st.text(texto_corto)

            prompt = build_prompt(pregunta, chunks)
            with st.spinner("Generando respuesta con el modelo local..."):
                try:
                    respuesta = generate_answer(prompt)
                except OllamaUnavailableError as exc:
                    fragmentos_preview = "\n\n".join(f"- {c.text[:200]}..." for c in chunks)
                    respuesta = (
                        "No se pudo generar la respuesta con Ollama.\n\n"
                        f"{exc}\n\n---\n**Fragmentos recuperados (modo solo-recuperación):**\n\n"
                        f"{fragmentos_preview}"
                    )
            st.markdown(respuesta)

    st.session_state.history.append({"role": "assistant", "content": respuesta})
