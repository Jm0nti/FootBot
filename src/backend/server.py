from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import logging
from typing import Optional

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelos de Pydantic
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    preferred_agent: Optional[str] = None

app = FastAPI(title="Soccer Agent Chatbot")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Ajustar en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INICIALIZACIÓN DEL BOT ---
_soccer_bot = None

def _init_bot_if_needed():
    global _soccer_bot
    if _soccer_bot is not None:
        return
    
    try:
        logger.info("⚽ Inicializando SoccerBot con LangGraph...")
        # Importamos la clase Wrapper que creamos en el Paso 1
        from app.Bots.soccer_bot import SoccerBot
        _soccer_bot = SoccerBot()
        logger.info("✅ SoccerBot listo.")
    except Exception as e:
        logger.exception("❌ Error inicializando SoccerBot: %s", e)
        raise

@app.on_event("startup")
def startup_event():
    _init_bot_if_needed()

# --- ENDPOINTS ---

@app.post("/api/chat")
def chat(req: ChatRequest):
    """Endpoint principal de chat conectado al Grafo."""
    _init_bot_if_needed()
    
    try:
        logger.info("/api/chat incoming message: %s", req.message)
        # Enviamos el mensaje al método .ask() de nuestra clase Wrapper
        response_data = _soccer_bot.ask(req.message)
        logger.info("/api/chat response agent=%s interaction_count=%s trace=%s",
                    response_data.get("agent_used"), _soccer_bot._interaction_count, response_data.get("trace"))
        
        return {
            "answer": response_data["answer"],
            "image": response_data.get("image"),
            "trace": response_data.get("trace"),
            "meta": {
                "agent": "LangGraph Multi-Agent",
                "interaction_count": _soccer_bot._interaction_count
            }
        }
    except Exception as e:
        logger.exception("Error en chat: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset")
def reset_conversation():
    """Reinicia la memoria del agente."""
    _init_bot_if_needed()
    _soccer_bot.clear_memory()
    logger.info("/api/reset called. Cleared bot memory.")
    return {"message": "Conversación reiniciada. Nueva sesión creada."}

@app.get("/health")
def health():
    return {"status": "ok", "bot_loaded": _soccer_bot is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)