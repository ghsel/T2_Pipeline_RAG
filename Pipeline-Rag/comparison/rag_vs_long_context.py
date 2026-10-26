"""
Unidad 2: comparación cuantitativa entre RAG y la estrategia de
"contexto largo" (pegar todo el documento en cada consulta), usando el
corpus realmente ingerido por este proyecto.

No se compara contra fine-tuning porque reentrenar un modelo requiere
GPU dedicada y no aplica a un proyecto que corre en una laptop; la
comparación conceptual contra fine-tuning queda documentada en el README
(tabla de la Unidad 2).
"""
import tiktoken

from rag.augmentation import build_prompt
from rag.retriever import Retriever
from rag.vector_store import VectorStore

# Encoding usada solo como referencia estándar para contar tokens de forma
# aproximada y consistente; no implica el uso de ningún servicio de OpenAI.
_ENCODING_NAME = "cl100k_base"


def count_tokens(text: str) -> int:
    encoder = tiktoken.get_encoding(_ENCODING_NAME)
    return len(encoder.encode(text))


def compare(query: str, top_k: int | None = None) -> dict:
    """Calcula cuántos tokens usaría el prompt en el enfoque RAG frente al
    enfoque de contexto largo, para la misma pregunta y el mismo corpus."""
    store = VectorStore()
    registros = store.get_all()

    if not registros:
        return {
            "error": "La base de conocimiento está vacía. Ejecuta primero: python cli.py ingest"
        }

    corpus_completo = "\n\n".join(r["text"] for r in registros)

    retriever = Retriever(store)
    chunks = retriever.search(query, top_k=top_k)

    prompt_rag = build_prompt(query, chunks)
    prompt_contexto_largo = (
        "INSTRUCCIONES\nResponde la pregunta usando el siguiente documento completo.\n\n"
        f"DOCUMENTO COMPLETO\n{corpus_completo}\n\nPREGUNTA\n{query}"
    )

    tokens_rag = count_tokens(prompt_rag)
    tokens_largo = count_tokens(prompt_contexto_largo)
    ahorro_pct = round((1 - tokens_rag / tokens_largo) * 100, 1) if tokens_largo else 0.0

    return {
        "query": query,
        "chunks_recuperados": len(chunks),
        "chunks_totales_en_corpus": len(registros),
        "tokens_prompt_rag": tokens_rag,
        "tokens_prompt_contexto_largo": tokens_largo,
        "ahorro_de_tokens_pct": ahorro_pct,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Compara el costo de contexto entre RAG y 'contexto largo' (Unidad 2)"
    )
    parser.add_argument("query", type=str, help="Pregunta a evaluar")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    print(json.dumps(compare(args.query, top_k=args.top_k), indent=2, ensure_ascii=False))
