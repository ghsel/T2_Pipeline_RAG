"""
RAG Research Studio — interfaz de investigación (Streamlit).

Tres zonas: panel izquierdo (espacios, fuentes, ajustes), conversación (centro) y panel de evidencia
(derecha, colapsable). La interfaz solo presenta: recuperación, evidencia, citas y estado de la respuesta
los calcula el backend (`rag.answering`).

Ejecutar con:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from app.ui import services, theme  # noqa: E402
from config.settings import settings  # noqa: E402
from rag.answering import answer_question  # noqa: E402
from rag.cache import ExactCache  # noqa: E402
from rag.retriever import Retriever  # noqa: E402
from rag.spaces import DEFAULT_SPACE  # noqa: E402

st.set_page_config(page_title="RAG Research Studio", page_icon="🔎", layout="wide")
st.markdown(theme.css(), unsafe_allow_html=True)


@st.cache_resource
def get_cache() -> ExactCache:
    return ExactCache()


@st.cache_resource
def get_backend(space: str):
    """Una conexión y un Retriever por espacio (el índice BM25 y el grafo se reutilizan entre consultas)."""
    store = services.open_store(space)
    return store, Retriever(store)


def _create_space_cb() -> None:
    try:
        st.session_state.space = services.create_space(st.session_state.new_space)
        st.session_state.space_msg = ("ok", f"Espacio «{st.session_state.space}» creado.")
    except ValueError as exc:
        st.session_state.space_msg = ("err", str(exc))


def _state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("selected", None)      # (índice de mensaje, nº de fragmento 1..k)
    st.session_state.setdefault("pending_delete", None)
    st.session_state.setdefault("upload_log", [])
    st.session_state.setdefault("processed_uploads", set())
    st.session_state.setdefault("space", DEFAULT_SPACE)


def _tag(texto: str) -> str:
    return f'<span class="tag">{texto}</span>'


def _badge(status: str) -> str:
    etiqueta, color = theme.STATUS.get(status, (status, "warn"))
    return f'<span class="badge {color}">{etiqueta}</span>'


_state()

# ----------------------------------------------------------------------------- panel izquierdo
with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-icon">▤</span><div><b>RAG Research Studio</b><br>'
                '<span class="ev-meta">Investigación documental con IA</span></div></div>', unsafe_allow_html=True)
    base = services.open_store(None)
    opciones = services.spaces(base)
    if st.session_state.space not in opciones:
        st.session_state.space = DEFAULT_SPACE
    st.selectbox("Espacio de investigación", opciones, key="space",
                 help="Cada espacio es un corpus aislado: las consultas no cruzan entre espacios.")
    with st.expander("Nuevo espacio"):
        st.text_input("Nombre (minúsculas, números, - y _)", key="new_space")
        st.button("Crear espacio", key="create_space", on_click=_create_space_cb)
        if st.session_state.get("space_msg"):
            nivel, msg = st.session_state.space_msg
            (st.success if nivel == "ok" else st.error)(msg)

    store, retriever = get_backend(st.session_state.space)

    cabecera_fuentes = st.empty()  # se rellena tras procesar las cargas, para no mostrar un conteo desactualizado
    filtro = st.text_input("Buscar documentos", key="src_filter", placeholder="Buscar documentos…",
                           label_visibility="collapsed").strip().lower()
    subidos = st.file_uploader("Añadir documentos (pdf, html, md)", accept_multiple_files=True, key="uploader")
    for f in subidos or []:
        clave = (st.session_state.space, f.name, f.size, f.file_id)
        if clave in st.session_state.processed_uploads:
            continue
        st.session_state.processed_uploads.add(clave)
        try:
            r = services.ingest_upload(store, f.name, f.getvalue())
            estado = "sin cambios" if r["unchanged"] else "indexado"
            st.session_state.upload_log.append(("ok", f"«{r['source']}»: {estado}, {r['n_chunks']} fragmentos."))
        except services.UploadError as exc:
            st.session_state.upload_log.append(("err", str(exc)))
    for nivel, msg in st.session_state.upload_log[-4:]:
        (st.success if nivel == "ok" else st.error)(msg)

    fuentes = store.list_sources()  # tras una posible carga
    cabecera_fuentes.markdown(f'<div class="side-head"><b>Fuentes</b><span class="ev-meta">{len(fuentes)} documentos</span></div>',
                              unsafe_allow_html=True)
    for s in fuentes:
        if filtro and filtro not in s["source"].lower():
            continue
        c1, c2 = st.columns([3, 1.3])
        c1.markdown(f'<div class="source-row"><span class="ico">▤</span><b>{s["source"]}</b><br><span class="ev-meta">{s["doc_type"]} · '
                    f'{s["n_chunks"]} fragmentos</span></div>', unsafe_allow_html=True)
        if c2.button("Eliminar", key=f"del_{s['source']}"):
            st.session_state.pending_delete = s["source"]
    pend = st.session_state.pending_delete
    if pend:
        n = next((s["n_chunks"] for s in fuentes if s["source"] == pend), 0)
        st.warning(f"¿Eliminar «{pend}» ({n} fragmentos)? Esta acción no se puede deshacer.")
        a, b = st.columns(2)
        if a.button("Confirmar eliminación", key="confirm_delete", type="primary"):
            services.delete_source(store, pend)
            st.session_state.pending_delete = None
            st.rerun()
        if b.button("Cancelar", key="cancel_delete"):
            st.session_state.pending_delete = None
            st.rerun()

    st.caption(f"{len(fuentes)} documentos · {store.count()} fragmentos")
    st.markdown("#### Configuración")
    top_k = st.slider("Fragmentos a recuperar (top-k)", 1, 10, settings.TOP_K)
    agentic = st.checkbox("Agentic RAG (reintenta si la evidencia es débil)", value=False)
    use_policy = st.checkbox("Política de evidencia", value=True,
                             help="Si está activa, el backend rechaza preguntas sin respaldo sin llamar al modelo.")
    with st.expander("Experimental"):
        use_rerank = st.checkbox("Reranker (cross-encoder local)", value=False,
                                 help=f"Modelo: {settings.RERANK_MODEL}. La primera vez descarga el modelo.")
        use_graph = st.checkbox("Expansión por relaciones (grafo)", value=False,
                                help="Añade fragmentos relacionados por entidades. No mejora de forma demostrada; ver informe.")
        use_cache = st.checkbox("Caché exacta", value=False,
                                help="Repite respuestas a consultas idénticas sin llamar al modelo.")
    st.caption(f"Embeddings: `{settings.EMBEDDING_MODEL}`")
    st.caption(f"Generación: `{settings.OLLAMA_MODEL}` (Ollama, local)")

retriever.use_graph = use_graph
retriever.use_rerank = use_rerank
cache = get_cache() if use_cache else None
store.cache = cache

# ----------------------------------------------------------------------------- cabecera
h1, h2 = st.columns([5, 1.6])
h1.markdown(f'<div class="studio-sub">Espacio «{st.session_state.space}» · {len(fuentes)} documentos · '
            f'{store.count()} fragmentos</div>', unsafe_allow_html=True)
show_ev = h2.toggle("Panel de evidencia", value=True, key="show_evidence")

centro, derecha = st.columns([3, 1.7]) if show_ev else (st.container(), None)

# ----------------------------------------------------------------------------- conversación
with centro:
    if not st.session_state.history:
        if store.count() == 0:
            st.info("Este espacio no tiene documentos. Añada archivos pdf, html o md en el panel izquierdo.")
        else:
            st.info("Haga una pregunta sobre los documentos de este espacio. Cada respuesta indica si está respaldada "
                    "y qué fragmentos la sustentan.")
    for idx, m in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.markdown(m["query"])
        r = m["result"]
        with st.chat_message("assistant", avatar="🔎"):
            st.markdown('<div class="answer-title"><span class="spark">✦</span>Respuesta</div>', unsafe_allow_html=True)
            tags = [_badge(r.status)]
            if r.cache_hits:
                tags.append(_tag("caché: " + ", ".join(r.cache_hits)))
            tags.append(_tag(m["method"]))
            tags.append(_tag(f"{r.timings.get('total_s', 0):.1f} s"))
            st.markdown(" ".join(tags), unsafe_allow_html=True)
            if r.status == "error":
                st.error(r.error or "Error técnico.")
            else:
                st.markdown(r.answer)
            if r.evidence is not None:
                st.caption(f"Evidencia: {r.evidence.status.value} — {r.evidence.reason}")
            nums = sorted({c["n"] for c in (r.citations.citations if r.citations else []) if c["valid"]})
            if nums:
                st.markdown("**Fuentes citadas** · pulse una para abrir el fragmento en el panel de evidencias")
                cols = st.columns(min(len(nums), 3))
                for i, n in enumerate(nums):
                    ch = r.chunks[n - 1] if 0 < n <= len(r.chunks) else None
                    etiqueta = f"[{n}] {ch.source[:24]}" + (f" p. {ch.page}" if ch and ch.page else "") if ch else f"[{n}]"
                    if cols[i % len(cols)].button(etiqueta, key=f"cite_{idx}_{n}"):
                        st.session_state.selected = (idx, n)
            for issue in (r.citations.issues if r.citations else []):
                st.caption(f"⚠ {issue['type']}: {issue['detail']}")

# ----------------------------------------------------------------------------- panel de evidencia
if derecha is not None:
    with derecha:
        hist = st.session_state.history
        sel = st.session_state.selected
        n_ev = len(hist[sel[0] if sel and sel[0] < len(hist) else -1]["result"].chunks) if hist else 0
        st.markdown(f'<div class="ev-head"><h4>Evidencias</h4><span class="count-pill">{n_ev}</span></div>'
                    '<div class="ev-meta">Fragmentos relevantes para esta respuesta</div>', unsafe_allow_html=True)
        if not hist:
            st.caption("Aquí aparecerán los fragmentos recuperados de cada respuesta.")
        else:
            idx = sel[0] if sel and sel[0] < len(hist) else len(hist) - 1
            r = hist[idx]["result"]
            st.caption(f"Respuesta #{idx + 1} · método: {hist[idx]['method']}")
            if not r.chunks:
                st.caption("No se recuperaron fragmentos.")
            citados = {c["n"] for c in (r.citations.citations if r.citations else []) if c["valid"]}
            for n, c in enumerate(r.chunks, start=1):
                elegido = bool(sel and sel[0] == idx and sel[1] == n)
                parciales = " · ".join(f"{k} {v:.2f}" for k, v in c.scores.items() if isinstance(v, (int, float)))
                ubic = c.source + (f", p. {c.page}" if c.page else "")
                # Acordeón cerrado por defecto; la cita [n] abre solo su fragmento (el cambio de etiqueta reinicia su estado)
                etiqueta = f"{n} · {ubic}" + (" · citada" if n in citados else "") + (" ▸ seleccionada" if elegido else "")
                with st.expander(etiqueta, expanded=elegido):
                    st.markdown(
                        f'<div class="ev-card{" selected" if elegido else ""}">'
                        f'<span class="ev-meta">id {str(c.chunk_id)[:10]} · {c.method} · score {c.score:.3f}'
                        f'{" · " + parciales if parciales else ""}</span>'
                        f'<div class="ev-text">{c.text}</div>'
                        f'{"<span class=cited-tag>Citada en la respuesta</span>" if n in citados else ""}</div>',
                        unsafe_allow_html=True)
            gt = hist[idx].get("graph_trace")
            if gt and gt.added:
                with st.expander("Relaciones usadas por el grafo"):
                    st.write("Entidades de la pregunta:", gt.query_entities or "(ninguna)")
                    for a in gt.added:
                        st.write(f"+ {a['source']} vía {', '.join(a['via'])}")

# ----------------------------------------------------------------------------- entrada
pregunta = st.chat_input("Escriba su pregunta sobre los documentos del espacio…")
if pregunta:
    metodo = "agentic" if agentic else "híbrida"
    if use_rerank:
        metodo += " + reranker"
    elif use_graph:
        metodo += " + grafo"
    with st.spinner("Recuperando evidencia y generando respuesta (el modelo local puede tardar)…"):
        resultado = answer_question(pregunta, retriever, top_k=top_k, agentic=agentic, use_policy=use_policy,
                                    prompt_version="v2", cache=cache)
    st.session_state.history.append({"query": pregunta, "result": resultado, "method": metodo,
                                     "graph_trace": retriever.last_graph_trace if use_graph else None})
    st.session_state.selected = (len(st.session_state.history) - 1, 0)  # respuesta nueva visible, ningún fragmento abierto
    st.rerun()
