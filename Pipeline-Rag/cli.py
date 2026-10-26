"""
CLI principal del proyecto RAG aplicado.

Comandos:
    python cli.py ingest [--source data/raw] [--reset] [--space nombre]
    python cli.py ask "pregunta" [--top-k 4] [--agentic] [--graph] [--rerank] [--no-generate] [--no-policy] [--prompt v1|v2]
    python cli.py spaces [--delete nombre]
    python cli.py compare "pregunta" [--top-k 4]
"""
import argparse
import json

from comparison.rag_vs_long_context import compare as compare_costs
from config.settings import settings
from ingestion.pipeline import run_pipeline
from rag.answering import answer_question
from rag.augmentation import build_prompt
from rag.retriever import Retriever
from rag.vector_store import VectorStore


def cmd_ingest(args: argparse.Namespace) -> None:
    resumen = run_pipeline(source_dir=args.source, reset=args.reset, space=args.space)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


def _print_chunks(chunks) -> None:
    print(f"\n--- {len(chunks)} fragmento(s) recuperado(s) ---")
    for i, c in enumerate(chunks, start=1):
        ubicacion = c.source + (f" (p.{c.page})" if c.page else "")
        print(f"  [{i}] {ubicacion} | método={c.method} | score={c.score:.3f} | id={c.chunk_id}")


def cmd_ask(args: argparse.Namespace) -> None:
    store = VectorStore(space=args.space)
    if store.count() == 0:
        print("La base de conocimiento está vacía. Ejecuta primero: python cli.py ingest")
        return

    retriever = Retriever(store, use_graph=True if args.graph else None, use_rerank=True if args.rerank else None)
    version = args.prompt or settings.PROMPT_VERSION

    if args.no_generate:
        if args.agentic:
            resultado = retriever.agentic_search(args.query, top_k=args.top_k)
            chunks = resultado.chunks
            print("--- Trazabilidad de recuperación (Agentic RAG) ---")
            for ronda in resultado.rounds:
                print(
                    f"  Ronda {ronda['round']}: top_k={ronda['top_k']}, "
                    f"score_promedio={ronda['avg_score']}, resultados={ronda['n_results']}"
                )
        else:
            chunks = retriever.search(args.query, top_k=args.top_k)
        _print_chunks(chunks)
        print("\n--- Prompt aumentado (modo solo-recuperación) ---")
        print(build_prompt(args.query, chunks, version=version))
        return

    if args.graph and retriever.last_graph_trace:
        _print_graph_trace(retriever.last_graph_trace)
    resultado = answer_question(
        args.query, retriever, top_k=args.top_k, agentic=args.agentic,
        use_policy=False if args.no_policy else None, prompt_version=version,
    )
    if resultado.rounds:
        print("--- Trazabilidad de recuperación (Agentic RAG) ---")
        for ronda in resultado.rounds:
            print(
                f"  Ronda {ronda['round']}: top_k={ronda['top_k']}, "
                f"score_promedio={ronda['avg_score']}, resultados={ronda['n_results']}"
            )
    _print_chunks(resultado.chunks)
    ev = resultado.evidence
    print(f"\n--- Evidencia: {ev.status.value} ({ev.reason}) ---")
    print(f"--- Estado de la respuesta: {resultado.status.upper()} "
          f"(política={'activa' if resultado.policy_enabled else 'desactivada'}, prompt {resultado.prompt_version}) ---")
    if resultado.status == "error":
        print(f"\n{resultado.error}")
        print("\nMostrando el prompt aumentado como alternativa (modo solo-recuperación):\n")
        print(build_prompt(args.query, resultado.chunks, version=version))
        return
    print(f"\n{resultado.answer}")
    if resultado.citations and resultado.citations.citations:
        print("\n--- Citas verificadas ---")
        for c in resultado.citations.citations:
            destino = f"{c['source']}" + (f" p.{c['page']}" if c["page"] else "") if c["valid"] else "NO EXISTE"
            print(f"  {c['marker']} -> {destino}")
    for issue in (resultado.citations.issues if resultado.citations else []):
        print(f"  [{issue['severity']}] {issue['type']}: {issue['detail']}")
    print(f"\n(tiempos: {resultado.timings})")

def _print_graph_trace(t) -> None:
    print("\n--- Expansión por relaciones (grafo) ---")
    print(f"  Entidades en la pregunta: {t.query_entities or '(ninguna)'}")
    print(f"  Entidades semilla: {t.seed_entities}")
    print(f"  Alcanzadas (entidad: saltos): {t.reached}")
    for a in t.added:
        print(f"  + {a['source']} [{a['chunk_id'][:8]}] puntaje={a['score']} vía {a['via']}")
    if t.note:
        print(f"  {t.note}")


def cmd_graph(args: argparse.Namespace) -> None:
    store = VectorStore(space=args.space)
    if store.count() == 0:
        print("La base de conocimiento está vacía. Ejecuta primero: python cli.py ingest")
        return
    retriever = Retriever(store)
    grafo = retriever.get_graph()
    print(json.dumps(grafo.stats(), indent=2, ensure_ascii=False))
    if args.query:
        base = retriever.hybrid_search(args.query, args.top_k or settings.TOP_K)
        print("\n--- Recuperación base (híbrida) ---")
        _print_chunks(base)
        final = retriever.graph_search(args.query, args.top_k or settings.TOP_K, normalize=args.normalize)
        _print_graph_trace(retriever.last_graph_trace)
        print("\n--- Contexto final con grafo ---")
        _print_chunks(final)


def cmd_spaces(args: argparse.Namespace) -> None:
    from rag.spaces import delete_space, list_spaces

    client = VectorStore().client
    if args.delete:
        ok = delete_space(client, args.delete)
        print(f"Espacio '{args.delete}' " + ("eliminado." if ok else "no existe."))
    for e in list_spaces(client):
        print(f"  {e['space']:<20} {e['n_chunks']:>5} chunk(s)  (colección {e['collection']})")


def cmd_compare(args: argparse.Namespace) -> None:
    resultado = compare_costs(args.query, top_k=args.top_k)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sistema RAG aplicado — CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_ingest = subparsers.add_parser("ingest", help="Ejecuta el pipeline de ingesta de documentos")
    p_ingest.add_argument("--source", type=str, default=None, help="Carpeta de documentos (default: data/raw)")
    p_ingest.add_argument("--reset", action="store_true", help="Reinicia la colección antes de ingerir")
    p_ingest.add_argument("--space", type=str, default=None, help="Espacio de investigación (default: 'default')")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = subparsers.add_parser("ask", help="Realiza una pregunta al sistema RAG")
    p_ask.add_argument("query", type=str, help="Pregunta en lenguaje natural")
    p_ask.add_argument("--top-k", type=int, default=None, help="Número de fragmentos a recuperar")
    p_ask.add_argument("--agentic", action="store_true", help="Usa el modo Agentic RAG (recuperación iterativa)")
    p_ask.add_argument(
        "--no-generate", action="store_true", help="Solo recupera y muestra el prompt, sin llamar al LLM"
    )
    p_ask.add_argument("--no-policy", action="store_true", help="Desactiva la política de evidencia (siempre llama al LLM)")
    p_ask.add_argument("--prompt", choices=["v1", "v2"], default=None, help="Versión del prompt (v1 original, v2 con citas [n])")
    p_ask.add_argument("--space", type=str, default=None, help="Espacio de investigación (default: 'default')")
    p_ask.add_argument("--graph", action="store_true", help="Activa la expansión por relaciones (grafo, experimental)")
    p_ask.add_argument("--rerank", action="store_true", help="Reordena los candidatos con un cross-encoder local (experimental)")
    p_ask.set_defaults(func=cmd_ask)

    p_graph = subparsers.add_parser("graph", help="Inspecciona el grafo de entidades y la expansión de una consulta")
    p_graph.add_argument("query", nargs="?", default=None, help="Consulta para ver la expansión (opcional)")
    p_graph.add_argument("--top-k", type=int, default=None)
    p_graph.add_argument("--space", type=str, default=None)
    p_graph.add_argument("--normalize", action="store_true", help="Normaliza el puntaje por nº de entidades del chunk")
    p_graph.set_defaults(func=cmd_graph)

    p_spaces = subparsers.add_parser("spaces", help="Lista o elimina espacios de investigación")
    p_spaces.add_argument("--delete", type=str, default=None, help="Elimina el espacio indicado por completo")
    p_spaces.set_defaults(func=cmd_spaces)

    p_compare = subparsers.add_parser(
        "compare", help="Compara el costo de contexto: RAG vs. contexto largo (Unidad 2)"
    )
    p_compare.add_argument("query", type=str)
    p_compare.add_argument("--top-k", type=int, default=None)
    p_compare.set_defaults(func=cmd_compare)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
