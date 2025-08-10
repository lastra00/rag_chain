# RAG Chain (Contrato de Arriendo)

API de RAG (Retrieval-Augmented Generation) para consultar un contrato de arriendo. Incluye servidor FastAPI con LangServe, contenedor Docker, `docker-compose`, y despliegue en Fly.io.

## Estructura

- `app/`
  - `rag_chain.py`: servidor FastAPI + LangServe. Expone la cadena RAG en `/rag_chain` y un health check en `/health`.
  - `__init__.py`
- `RAG_Contrato_Arriendo.ipynb`: notebook que extrae el texto del PDF (incluye fallback OCR), crea chunks y carga/crea la colección en Qdrant.
- `Contrato de arriendo departamento notariado - Pablo Lastra.PDF`: PDF fuente (opcional; puedes ignorarlo en Git).
- `requirements.txt`: dependencias para el servidor.
- `Dockerfile`: imagen del servidor (`uvicorn app.rag_chain:app`).
- `docker-compose.yml`: orquesta el servicio local en `:8000` y carga `.env`.
- `fly.toml`: configuración para deploy en Fly.io.

## Variables de entorno

Define estas claves (sin comillas) en tu `.env`:

- `OPENAI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`

El servidor carga variables con `python-dotenv`. En Docker Compose se lee el `.env` del proyecto; en Fly.io se configuran como secrets.

## Ejecutar local

- Python directo:
  - `python app/rag_chain.py`
  - Abre `http://localhost:8000/rag_chain/playground/`

- Docker Compose:
  - `docker compose up --build`
  - Abre `http://localhost:8000/rag_chain/playground/`

Endpoints relevantes:

- Playground: `/rag_chain/playground/`
- API (LangServe): `/rag_chain/`
- Salud: `/health`
- OpenAPI: `/docs` (nota: con Pydantic v2, ciertos endpoints de LangServe pueden no aparecer en Docs; la API y el Playground funcionan igualmente)

## Vector store (Qdrant)

La API asume que existe una colección Qdrant llamada `contrato_arriendo_pablo` con los chunks del contrato. Si no existe, crea/indiza desde el notebook `RAG_Contrato_Arriendo.ipynb` (sección de conexión a Qdrant y carga de documentos).

## Despliegue en Fly.io

Requisitos: `flyctl` instalado y sesión iniciada.

1. Secrets (sin comillas):
   - `flyctl secrets set -a rag-chain OPENAI_API_KEY=sk-...`
   - `flyctl secrets set -a rag-chain QDRANT_URL=https://...`
   - `flyctl secrets set -a rag-chain QDRANT_API_KEY=...`
   - Alternativa: `flyctl secrets import -a rag-chain < /ruta/a/.env`
2. Deploy: `flyctl deploy --remote-only`
3. Verifica:
   - Health: `https://rag-chain.fly.dev/health`
   - Playground: `https://rag-chain.fly.dev/rag_chain/playground/`

Notas Fly.io:

- `fly.toml` usa `internal_port = 8000` y el contenedor escucha en `0.0.0.0:8000`.
- Auto-escalado por inactividad está activado: la primera petición tras un rato puede demorar (cold start). Para mantener siempre encendido: en `[http_service]` usa `auto_stop_machines = 'off'` y `min_machines_running = 1`.

## Solución de problemas

- 401 Invalid OpenAI API key en `/health` o en respuestas:
  - Asegúrate de no poner comillas en el valor del secret (`OPENAI_API_KEY=sk-...`). Si importaste desde `.env` con comillas, vuelve a setear el secret sin comillas.

- 502 en Fly.io al abrir el Playground:
  - Comprueba `https://rag-chain.fly.dev/health`.
  - Revisa logs: `flyctl logs -a rag-chain`.
  - Verifica secrets y conectividad a Qdrant.

- `/docs` muestra error o no lista endpoints de LangServe:
  - Es una limitación conocida con Pydantic v2. La API y el Playground siguen funcionando.

- Error de streaming: instalar `sse-starlette` (ya incluido en `requirements.txt`).


```