"""Migracion no destructiva de Chroma/Google a Chroma/OpenAI.

El indice anterior se lee únicamente; la nueva colección se construye en
una carpeta temporal y se publica solo al terminar correctamente.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

LEGACY_PATH = Path(os.getenv("PDOT_LEGACY_DB_PATH", "db_pdot"))
NEW_PATH = Path(os.getenv("PDOT_OPENAI_DB_PATH", "db_pdot_openai"))
MODEL = os.getenv("PDOT_EMBEDDING_MODEL", "text-embedding-3-small")
MARKER = "migration-complete.json"


def obtener_indice(embeddings: OpenAIEmbeddings) -> Chroma:
    if not LEGACY_PATH.is_dir():
        raise RuntimeError(f"No existe el indice previo: {LEGACY_PATH}")

    if NEW_PATH.exists():
        if not (NEW_PATH / MARKER).is_file():
            raise RuntimeError(
                f"El indice de destino {NEW_PATH} existe sin confirmacion de migracion. "
                "No se modificara automaticamente. Revise el directorio."
            )
        metadata = json.loads((NEW_PATH / MARKER).read_text(encoding="utf-8"))
        if metadata.get("embedding_model") != MODEL:
            raise RuntimeError("La coleccion existente se creo con otro modelo de embeddings")
        return Chroma(persist_directory=str(NEW_PATH), embedding_function=embeddings)

    # Leer DOCUMENTOS y metadatos del indice viejo, sin consultar Google y sin
    # generar embeddings de Google. El antiguo almacenamiento NO se modifica.
    legacy = Chroma(persist_directory=str(LEGACY_PATH))
    previous = legacy.get(include=["documents", "metadatas"])
    ids = previous.get("ids") or []
    docs = previous.get("documents") or []
    metas = previous.get("metadatas") or []
    if not ids or len(ids) != len(docs) or len(ids) != len(metas):
        raise RuntimeError("Indice antiguo vacio o inconsistente; migracion cancelada")

    tmp = NEW_PATH.with_name(NEW_PATH.name + ".building")
    if tmp.exists():
        # Solo el directorio temporal reservado para una migracion interrumpida.
        shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)

    try:
        new_db = Chroma(persist_directory=str(tmp), embedding_function=embeddings)
        for i in range(0, len(ids), 50):
            chunk_ids = ids[i:i+50]
            chunk_docs = docs[i:i+50]
            chunk_metas = [(m or {}) for m in metas[i:i+50]]
            if any(not isinstance(d, str) or not d.strip() for d in chunk_docs):
                raise RuntimeError("El indice contiene un fragmento vacio; revisar antes de migrar")
            new_db.add_texts(texts=chunk_docs, metadatas=chunk_metas, ids=chunk_ids)
        if new_db._collection.count() != len(ids):
            raise RuntimeError("Conteo de fragmentos distinto tras indexacion")
        (tmp / MARKER).write_text(json.dumps({
            "embedding_model": MODEL,
            "documents": len(ids),
            "legacy_path": str(LEGACY_PATH),
        }), encoding="utf-8")
        tmp.rename(NEW_PATH)
        return Chroma(persist_directory=str(NEW_PATH), embedding_function=embeddings)
    except Exception:
        # Preservar evidencia para diagnostico; nunca borra el indice anterior.
        raise
