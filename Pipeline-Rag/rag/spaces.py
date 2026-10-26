"""
Espacios de investigación: cada espacio es una colección Chroma independiente, de modo
que una consulta nunca mezcla documentos de otro corpus.

El espacio "default" usa la colección original (`settings.COLLECTION_NAME`), por lo que
los datos ya ingeridos siguen siendo accesibles sin migración.
"""
import re

from config.settings import settings

DEFAULT_SPACE = "default"
_PREFIX = "space_"
_VALID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")


def validate_space_name(space: str) -> str:
    """Normaliza a minúsculas y valida (letras, números, '_' y '-'; máx. 48 caracteres)."""
    nombre = (space or "").strip().lower()
    if not _VALID.match(nombre):
        raise ValueError(
            f"Nombre de espacio inválido: '{space}'. Use minúsculas, números, '_' o '-' "
            "(máx. 48 caracteres, debe empezar con letra o número)."
        )
    return nombre


def collection_name_for(space: str | None) -> str:
    nombre = validate_space_name(space or DEFAULT_SPACE)
    return settings.COLLECTION_NAME if nombre == DEFAULT_SPACE else f"{_PREFIX}{nombre}"


def list_spaces(client) -> list[dict]:
    """Espacios existentes en el cliente Chroma: [{space, collection, n_chunks}], por nombre."""
    espacios = []
    for col in client.list_collections():
        nombre = getattr(col, "name", col)
        if nombre == settings.COLLECTION_NAME:
            espacio = DEFAULT_SPACE
        elif nombre.startswith(_PREFIX):
            espacio = nombre[len(_PREFIX):]
        else:
            continue
        n = client.get_collection(nombre).count()
        espacios.append({"space": espacio, "collection": nombre, "n_chunks": n})
    return sorted(espacios, key=lambda e: e["space"])


def delete_space(client, space: str) -> bool:
    """Elimina por completo la colección del espacio. Devuelve False si no existía."""
    nombre = collection_name_for(space)
    if nombre not in {getattr(c, "name", c) for c in client.list_collections()}:
        return False
    client.delete_collection(nombre)
    return True
