from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MODELOS DE PYDANTIC
# ============================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    preferred_agent: Optional[str] = None

class FormationData(BaseModel):
    """Modelo para datos de formación táctica"""
    team_name: str
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    success: bool = False

class ChatResponse(BaseModel):
    """Respuesta del chat con soporte para formaciones"""
    answer: str
    image: Optional[str] = None  # Para backward compatibility
    formation: Optional[FormationData] = None  # NUEVO: datos de formación
    trace: Optional[list] = None
    meta: Dict[str, Any] = {}

# ============================================================================
# CONFIGURACIÓN DE FASTAPI
# ============================================================================

app = FastAPI(title="Soccer Agent Chatbot")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajustar en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar carpeta de assets como archivos estáticos (NUEVO)
# Esto permite servir imágenes directamente vía URL
assets_path = Path("assets")
if assets_path.exists():
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")
    logger.info("📁 Assets folder mounted at /assets")
else:
    logger.warning("⚠️ Assets folder not found. Creating it...")
    assets_path.mkdir(parents=True, exist_ok=True)
    (assets_path / "formations").mkdir(exist_ok=True)

# ============================================================================
# INICIALIZACIÓN DEL BOT
# ============================================================================

_soccer_bot = None

def _init_bot_if_needed():
    global _soccer_bot
    if _soccer_bot is not None:
        return
    
    try:
        logger.info("⚽ Inicializando SoccerBot con LangGraph...")
        from app.Bots.soccer_bot import SoccerBot
        _soccer_bot = SoccerBot()
        logger.info("✅ SoccerBot listo.")
    except Exception as e:
        logger.exception("❌ Error inicializando SoccerBot: %s", e)
        raise

@app.on_event("startup")
def startup_event():
    _init_bot_if_needed()

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Endpoint principal de chat conectado al Grafo.
    Ahora soporta imágenes de formaciones tácticas.
    """
    _init_bot_if_needed()
    
    try:
        logger.info("[/api/chat] Incoming message: %s", req.message)
        
        # Enviar mensaje al bot
        response_data = _soccer_bot.ask(req.message)
        
        logger.info("[/api/chat] Response - agent=%s, interaction_count=%s, trace=%s",
                    response_data.get("agent_used"), 
                    _soccer_bot._interaction_count, 
                    response_data.get("trace"))
        
        # ========== PREPARAR RESPUESTA BASE ==========
        chat_response = {
            "answer": response_data["answer"],
            "image": response_data.get("image"),  # Backward compatibility
            "trace": response_data.get("trace"),
            "formation": None,  # Se llenará si hay formación
            "meta": {
                "agent": "LangGraph Multi-Agent",
                "interaction_count": _soccer_bot._interaction_count,
                "agent_used": response_data.get("agent_used")
            }
        }
        
        # ========== MANEJO DE FORMACIONES (NUEVO) ==========
        # Si el bot retornó datos de formación, incluirlos en la respuesta
        if "formation_data" in response_data and response_data["formation_data"]:
            formation_data = response_data["formation_data"]
            
            # Validar que sea una formación exitosa
            if formation_data.get("success") and formation_data.get("image_base64"):
                chat_response["formation"] = {
                    "team_name": formation_data.get("team_name"),
                    "image_base64": formation_data.get("image_base64"),
                    "image_url": formation_data.get("image_url"),
                    "success": True
                }
                logger.info("[/api/chat] 📋 Enviando formación para: %s", 
                          formation_data.get("team_name"))
            else:
                # Formación solicitada pero no encontrada
                logger.warning("[/api/chat] ⚠️ Formación no encontrada para: %s", 
                             formation_data.get("team_name"))
        
        return chat_response
    
    except Exception as e:
        logger.exception("[/api/chat] ❌ Error en chat: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reset")
def reset_conversation():
    """Reinicia la memoria del agente."""
    _init_bot_if_needed()
    _soccer_bot.clear_memory()
    logger.info("[/api/reset] Memoria del bot limpiada.")
    return {"message": "Conversación reiniciada. Nueva sesión creada."}


@app.get("/health")
def health():
    """Health check del servidor."""
    return {
        "status": "ok",
        "bot_loaded": _soccer_bot is not None,
        "assets_mounted": (Path("assets") / "formations").exists()
    }


@app.get("/api/formations/{team_name}")
def get_formation_image(team_name: str):
    """
    Endpoint opcional para obtener imágenes de formación directamente.
    
    Uso: GET /api/formations/barcelona
    Retorna: Imagen PNG
    """
    try:
        formations_dir = Path("assets/formations")
        
        # Buscar archivo
        team_clean = team_name.lower().replace(" ", "_")
        possible_files = [
            f"{team_clean}_formation.png",
            f"{team_clean}.png"
        ]
        
        for filename in possible_files:
            file_path = formations_dir / filename
            if file_path.exists():
                logger.info("[/api/formations] Serving: %s", filename)
                return FileResponse(
                    file_path,
                    media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"}
                )
        
        logger.warning("[/api/formations] Not found: %s", team_name)
        raise HTTPException(
            status_code=404, 
            detail=f"Formación no encontrada para {team_name}"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[/api/formations] Error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/formations")
def list_formations():
    """
    Lista todas las formaciones disponibles.
    
    Uso: GET /api/formations
    Retorna: Lista de equipos con formaciones disponibles
    """
    try:
        formations_dir = Path("assets/formations")
        
        if not formations_dir.exists():
            return {"formations": []}
        
        formations = []
        for file_path in formations_dir.glob("*.png"):
            team_name = file_path.stem.replace("_formation", "").replace("_", " ").title()
            formations.append({
                "team_name": team_name,
                "filename": file_path.name,
                "url": f"/assets/formations/{file_path.name}"
            })
        
        logger.info("[/api/formations] Listed %d formations", len(formations))
        return {"formations": formations, "count": len(formations)}
    
    except Exception as e:
        logger.exception("[/api/formations] Error listing: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ENDPOINT DE DEBUG (Opcional - útil para desarrollo)
# ============================================================================

@app.get("/api/debug/state")
def debug_state():
    """
    Endpoint de debug para ver el estado interno del bot.
    REMOVER EN PRODUCCIÓN.
    """
    _init_bot_if_needed()
    
    return {
        "bot_loaded": _soccer_bot is not None,
        "interaction_count": _soccer_bot._interaction_count if _soccer_bot else 0,
        "formations_available": len(list(Path("assets/formations").glob("*.png"))) 
                                if Path("assets/formations").exists() else 0
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Verificar estructura de carpetas
    print("\n" + "=" * 80)
    print("🚀 SOCCER AGENT CHATBOT - SERVER")
    print("=" * 80)
    print(f"📁 Assets folder: {Path('assets').absolute()}")
    print(f"📋 Formations folder: {Path('assets/formations').absolute()}")
    
    formations_path = Path("assets/formations")
    if formations_path.exists():
        num_formations = len(list(formations_path.glob("*.png")))
        print(f"✅ {num_formations} formaciones encontradas")
    else:
        print("⚠️ No se encontró la carpeta de formaciones")
    
    print("=" * 80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)