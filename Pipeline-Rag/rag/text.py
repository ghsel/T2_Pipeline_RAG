"""
Normalización y tokenización de texto para la búsqueda léxica (BM25).

La tokenización original (`texto.lower().split()`) dejaba la puntuación pegada
a las palabras ("días,", "reembolso.") y distinguía "garantía" de "garantia".
Aquí se normaliza de forma simétrica para documentos y consultas:

  - minúsculas y sin tildes (NFKD);
  - las cifras se conservan completas ("4,5", "10.000", "0,05"), porque en
    documentos financieros el número exacto es la señal léxica más útil;
  - se descartan palabras vacías muy frecuentes del español.
"""
import re
import unicodedata

# Cifra con separadores decimales/millares, o palabra alfanumérica.
_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*|\w+")

STOPWORDS_ES = frozenset(
    """
    a al algo ante con contra cual cuales cuando de del desde donde e el ella ellas ello
    ellos en entre era es esa ese eso esta estas este esto estos fue ha han hay la las le
    les lo los me mi mis muy ni no nos o para pero por porque que quien se si sin sobre su
    sus te tu tus un una uno unos y ya
    """.split()
)


def strip_accents(text: str) -> str:
    """Elimina tildes y diacríticos ("garantía" -> "garantia")."""
    descompuesto = unicodedata.normalize("NFKD", text)
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Convierte un texto en tokens normalizados para BM25."""
    tokens = _TOKEN_RE.findall(strip_accents(text.lower()))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_ES]
    return tokens
