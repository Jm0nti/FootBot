# FootBot Multi-Agent Chatbot

Proyecto que integra un sistema multiagente (LangGraph + LangChain wrappers) para responder preguntas relacionadas con el fútbol: estadísticas de jugadores y equipos, formaciones, historia y noticias en tiempo real.

**Propósito**
- Orquestar varios agentes (clasificador, SQL-statistics agent, RAG agent, web-search agent, formation agent y crítico) mediante un grafo (`langgraph`) para proporcionar respuestas especializadas.
- Exponer una API HTTP (FastAPI) para interacción desde el frontend o clientes externos.
- Proveer trazas (checkpoints) y logging para depuración del enrutamiento entre agentes y ejecución de herramientas.

**Estructura del repositorio**
- `src/backend/` - Backend FastAPI y lógica del bot
  - `server.py` - Entrypoint del servidor y endpoints `/api/chat`, `/api/reset`, `/health`.
  - `app/Bots/soccer_bot.py` - Implementación del grafo, nodos (agentes), herramientas y wrapper `SoccerBot`.
  - `pyproject.toml`, `requirements.txt` - Dependencias (revisar y usar el método de instalación elegido).
  - `assets/` - Recursos estáticos (por ejemplo `assets/formations/` para imágenes de formaciones).
- `src/frontend/` - Interfaz web (Vite + React)
  - `src/` - Código fuente del frontend (componentes y assets)
- `exploratorio/` - Scripts y datos auxiliares usados localmente (CSV, prototipos)

**Características principales**
- Multi-agente en grafo (classifier → agente específico → critic)
- Herramientas definidas como `@tool` (SQL executor, FAISS retriever, web search, formation image lookup)
- Logging y `trace` agregado a `AgentState` y retornado en `/api/chat` para debug: permite ver la secuencia de nodos visitados y llamadas a herramientas.

Prerequisitos
- Python 3.10+ (recomendado)
- Node.js 16+ (para el frontend)

Variables de entorno (archivo `.env`)
- `OPENAI_API_KEY` 
- `PERPLEXITY_API_KEY` 
- `GOOGLE_API_KEY` 
- `GROQ_API_KEY` 

Instalación (backend)
- Crear entorno virtual y activar (PowerShell):

```powershell
cd 'c:\Users\juanl\Documents\SistemaMultiagente_PLN\src\backend'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Ejecución (backend)
- Desde `src/backend`:

```powershell
uvicorn server:app --reload
```

- Endpoints principales:
  - `POST /api/chat` → JSON { "message": "tu pregunta" }
    - Respuesta incluye `answer`, `image` (si aplica), `trace` y `meta`.
  - `POST /api/reset` → Reinicia la memoria de la sesión/agent
  - `GET /health` → Estado básico

Ejecución (frontend)
- Desde `src/frontend`:

```powershell
cd 'c:\Users\juanl\Documents\SistemaMultiagente_PLN\src\frontend'
npm install
npm run dev
```

Depuración y logs
- El backend usa `logging` con nivel `INFO` por defecto (configurado en `server.py`).
- En `soccer_bot.py` se añadieron logs y un campo `trace` dentro del estado del grafo que contiene la secuencia de nodos visitados. Ejemplo de salida esperada en consola:
  - `[classifier] Clasificado como: formation`
  - `[formation] Equipo extraído: Barcelona`
  - `[formation_image_tool] Imagen encontrada: Barcelona_Formation.png`
  - `[SoccerBot.ask] Respuesta generada. agent=formation, trace=['classifier','formation']`

Respuesta `POST /api/chat` (ejemplo)
```json
{
  "answer": "Formación táctica del Barcelona: ...",
  "image": "/assets/formations/Barcelona_Formation.png",
  "trace": ["classifier","formation"],
  "meta": { "agent": "LangGraph Multi-Agent", "interaction_count": 3 }
}
```
