"""
Generation (Unidad 1): el LLM redacta la respuesta final, fundamentada en
el prompt aumentado construido por `rag/augmentation.py`.

Por defecto, la generación corre 100% local mediante **Ollama**
(https://ollama.com), sirviendo modelos open-weight (Llama 3.2, Qwen2.5,
Mistral, etc.) sin ninguna llamada a una API de pago ni consumo de
créditos/tokens facturables. El generador sigue siendo "un LLM estándar";
lo único que cambia es lo que recibe como entrada (tal como señala el
material del curso).
"""
import ollama

from config.settings import settings


class OllamaUnavailableError(RuntimeError):
    """Se lanza cuando Ollama no está corriendo o el modelo no está disponible."""


def generate_answer(prompt: str) -> str:
    """Envía el prompt aumentado al modelo local vía Ollama y devuelve la respuesta."""
    try:
        respuesta = ollama.chat(
            model=settings.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": settings.GENERATION_TEMPERATURE},
        )
        return respuesta["message"]["content"]
    except Exception as exc:  # noqa: BLE001 — se traduce a un error propio y explicativo
        raise OllamaUnavailableError(
            f"No se pudo generar una respuesta con Ollama (modelo '{settings.OLLAMA_MODEL}').\n"
            f"¿Está Ollama corriendo? Prueba: 'ollama serve' y 'ollama pull "
            f"{settings.OLLAMA_MODEL}'.\nDetalle técnico: {exc}"
        ) from exc
