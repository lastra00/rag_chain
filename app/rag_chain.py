from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langserve import add_routes

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda


def _validate_environment() -> None:
    load_dotenv()
    required_env_vars: List[str] = ["OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
    missing: List[str] = [key for key in required_env_vars if not os.getenv(key)]
    if missing:
        raise RuntimeError(
            f"Faltan variables de entorno requeridas: {missing}. Configura tu archivo .env."
        )


def build_contract_rag_chain():
    _validate_environment()

    # Modelos de OpenAI
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=512,
    )
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

    # Cliente Qdrant y vector store existente (colección creada desde el notebook)
    collection_name = "contrato_arriendo_pablo"
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    # Aseguramos que la colección exista; si no, pedimos crearla desde el notebook
    collections = {c.name for c in qdrant_client.get_collections().collections}
    if collection_name not in collections:
        raise RuntimeError(
            f"La colección '{collection_name}' no existe en Qdrant. "
            "Primero crea/indiza los documentos desde el notebook RAG_Contrato_Arriendo.ipynb."
        )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3, "score_threshold": 0.1}
    )

    def format_contract_docs(docs):
        formatted: List[str] = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "N/A")
            text = doc.page_content
            formatted.append(f"[Fragmento {i} - Página {page}]\nContenido: {text}\n")
        return "\n" + ("=" * 50) + "\n" + "\n".join(formatted)

    contract_prompt = ChatPromptTemplate.from_template(
        """
## ROL
Eres un asistente legal especializado en contratos de arriendo en Chile.

## TAREA
Responde preguntas específicas sobre el contrato de arriendo basándote ÚNICAMENTE en la información proporcionada.

## INSTRUCCIONES:
1. Analiza cuidadosamente todos los fragmentos del contrato proporcionados
2. Responde SOLO con información que esté explícitamente en el contrato
3. Cita específicamente la página o sección donde encontraste la información
4. Si no encuentras información, indica claramente "Esta información no está disponible en las secciones analizadas del contrato"
5. Para temas legales complejos, sugiere consultar con un abogado
6. Estructura tu respuesta de manera clara y profesional
7. Incluye montos, fechas y datos específicos exactamente como aparecen en el contrato

## FRAGMENTOS DEL CONTRATO:
{context}

## PREGUNTA:
{question}

## RESPUESTA:
Según el contrato analizado:
"""
    )

    contract_rag_chain = (
        RunnableParallel(
            {"context": retriever | format_contract_docs, "question": RunnablePassthrough()}
        )
        | contract_prompt
        | llm
        | StrOutputParser()
    )

    return contract_rag_chain


# FastAPI + LangServe
app = FastAPI(
    title="rag_chain",
    version="1.0",
    description="RAG para contrato de arriendo (Qdrant + OpenAI)",
)

# Inicialización perezosa del RAG para evitar caída en arranque si faltan
# credenciales o Qdrant no está accesible
_cached_chain = None
_last_error: str | None = None


def _ensure_chain_initialized():
    global _cached_chain, _last_error
    if _cached_chain is None:
        try:
            _cached_chain = build_contract_rag_chain()
            _last_error = None
        except Exception as exc:  # noqa: BLE001
            _last_error = str(exc)
            _cached_chain = None


def _rag_invoke(user_input: str) -> str:
    _ensure_chain_initialized()
    if _cached_chain is None:
        return (
            "Servicio no disponible: "
            + (_last_error or "error desconocido inicializando la cadena")
        )
    try:
        return _cached_chain.invoke(user_input)
    except Exception as exc:  # noqa: BLE001
        return f"Error procesando la consulta: {exc}"


add_routes(
    app,
    RunnableLambda(_rag_invoke),
    path="/rag_chain",
    input_type=str,
    output_type=str,
)


@app.get("/health")
def health() -> dict:
    _ensure_chain_initialized()
    return {
        "status": "ok" if _cached_chain is not None else "degraded",
        "error": _last_error,
    }

# Ocultar endpoints de streaming y playground del esquema OpenAPI para evitar
# errores en /openapi.json en algunas combinaciones de versiones
for route in list(app.routes):
    try:
        path: str = getattr(route, "path", "")
        if path.endswith("/stream") or path.endswith("/stream_log") or "/playground" in path:
            if hasattr(route, "include_in_schema"):
                route.include_in_schema = False
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


