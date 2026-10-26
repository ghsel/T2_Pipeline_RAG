from rag.augmentation import build_prompt, format_context
from rag.vector_store import RetrievedChunk


def test_format_context_includes_source_and_page():
    chunks = [
        RetrievedChunk(text="El plazo es de 30 días.", source="politica.md", page=None, score=0.91),
        RetrievedChunk(text="Ver sección 4 de garantía.", source="manual.pdf", page=7, score=0.78),
    ]

    resultado = format_context(chunks)

    assert "politica.md" in resultado
    assert "manual.pdf, página 7" in resultado
    assert "El plazo es de 30 días." in resultado
    assert "0.91" in resultado


def test_build_prompt_has_three_required_sections():
    chunks = [RetrievedChunk(text="Texto relevante.", source="doc.md", page=None, score=0.5)]
    prompt = build_prompt("¿Cuál es el plazo?", chunks)

    assert "INSTRUCCIONES" in prompt
    assert "CONTEXTO RECUPERADO" in prompt
    assert "PREGUNTA DEL USUARIO" in prompt
    assert "¿Cuál es el plazo?" in prompt
    assert "Texto relevante." in prompt


def test_build_prompt_handles_empty_context_gracefully():
    prompt = build_prompt("¿Existe esta información?", [])
    assert "No se encontró ningún fragmento relevante" in prompt
    assert "¿Existe esta información?" in prompt
