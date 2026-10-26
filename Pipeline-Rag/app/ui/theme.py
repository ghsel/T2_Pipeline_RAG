"""Tokens de diseño y CSS de «RAG Research Studio». Un único bloque, sin lógica de negocio."""

TOKENS = {
    "primary": "#106CB7", "primary_dark": "#0B4F87", "primary_tint": "#E8F1F9",
    "bg": "#FFFFFF", "surface": "#F6F8FA", "text": "#1B2430", "text_muted": "#5B6675",
    "border": "#E2E6EB", "ok": "#1B7F51", "warn": "#9B661A", "err": "#C0392B",
}

# Estado de la respuesta -> (etiqueta visible, token de color)
STATUS = {
    "respaldada": ("Respaldada", "ok"),
    "parcial": ("Parcial", "warn"),
    "sin_evidencia": ("Evidencia insuficiente", "err"),
    "sin_documentos": ("Sin documentos", "err"),
    "error": ("Error técnico", "err"),
}


def css() -> str:
    t = TOKENS
    return f"""
<style>
:root {{ {' '.join(f'--{k.replace("_", "-")}: {v};' for k, v in t.items())} }}
.block-container {{ padding-top: 4rem; max-width: 1500px; }}
*:focus-visible {{ outline: 3px solid #106CB7 !important; outline-offset: 2px; }}
h1.studio-title {{ font-size: 1.45rem; margin: 0; color: var(--text); }}
.studio-sub {{ color: var(--text-muted); font-size: .9rem; margin-bottom: .6rem; }}
.badge {{ display: inline-block; padding: .12rem .6rem; border-radius: 999px; font-size: .78rem; font-weight: 600;
          color: #fff; margin-right: .4rem; }}
.badge.ok {{ background: var(--ok); }} .badge.warn {{ background: var(--warn); }} .badge.err {{ background: var(--err); }}
.tag {{ display: inline-block; padding: .08rem .5rem; border-radius: 6px; font-size: .75rem; margin-right: .3rem;
        background: var(--primary-tint); color: var(--primary-dark); border: 1px solid var(--border); }}
.ev-card {{ border: 1px solid var(--border); border-radius: 10px; padding: .6rem .8rem; margin-bottom: .6rem; background: var(--bg); }}
.ev-card.selected {{ border: 2px solid var(--primary); background: var(--primary-tint); }}
.ev-meta {{ color: var(--text-muted); font-size: .78rem; }}
.ev-text {{ font-size: .86rem; line-height: 1.4; white-space: pre-wrap; }}
.source-row {{ font-size: .88rem; background: var(--bg); border: 1px solid var(--border); border-radius: 10px;
               padding: .45rem .6rem; }}
.source-row .ico {{ color: var(--primary); margin-right: .4rem; }}
.brand {{ display: flex; gap: .6rem; align-items: center; margin-bottom: 1rem; }}
.brand .brand-icon {{ color: var(--primary); font-size: 1.5rem; }}
.side-head {{ display: flex; justify-content: space-between; align-items: baseline; margin: .4rem 0; }}
.side-head b {{ font-size: 1rem; }}
.ev-head {{ display: flex; justify-content: space-between; align-items: center; }}
.ev-head h4 {{ margin: 0; }}
.count-pill {{ background: var(--primary-tint); color: var(--primary-dark); border-radius: 999px; min-width: 1.9rem;
               text-align: center; font-weight: 700; padding: .15rem .5rem; }}
.ev-num {{ display: inline-block; width: 1.5rem; height: 1.5rem; line-height: 1.5rem; text-align: center; border-radius: 50%;
           background: var(--primary-tint); color: var(--primary-dark); font-weight: 700; font-size: .8rem; margin-right: .5rem; }}
.ev-card .ev-text {{ border-left: 3px solid var(--border); padding-left: .6rem; margin-top: .4rem; font-style: italic; }}
.cited-tag {{ display: inline-block; margin-top: .4rem; padding: .08rem .6rem; border-radius: 999px; font-size: .75rem;
              background: var(--primary-tint); color: var(--primary-dark); }}
.answer-title {{ font-size: 1.25rem; font-weight: 700; margin: .2rem 0 .4rem; color: var(--text); }}
.answer-title .spark {{ display: inline-block; width: 2rem; height: 2rem; line-height: 2rem; text-align: center;
                        border-radius: 50%; background: var(--primary-tint); color: var(--primary); margin-right: .5rem; }}
.stApp {{ background: #F3F6FA; }}
[data-testid="stChatMessage"] {{ background: var(--bg); border: 1px solid var(--border); border-radius: 14px; }}
[data-testid="stChatInputSubmitButton"] {{ background: var(--primary); color: #fff; border-radius: 50%; }}
section[data-testid="stSidebar"] {{ background: #EEF2F7; border-right: 1px solid var(--border); }}
.stButton > button[kind="primary"] {{ background: var(--primary); border-color: var(--primary); color: #fff; }}
.stButton > button[kind="primary"]:hover {{ background: var(--primary-dark); border-color: var(--primary-dark); }}
.stButton > button:focus-visible, a:focus-visible {{ outline: 3px solid var(--primary-dark); outline-offset: 2px; }}
</style>
"""
