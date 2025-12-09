import logging
from langchain_core.messages import SystemMessage
from app.Bots.models import llm_fast
from app.Bots.types import AgentState

logger = logging.getLogger(__name__)


def identity_node(state: AgentState) -> dict:
    """Responde preguntas sobre la identidad y capacidades del sistema"""
    identity_prompt = """Eres un asistente especializado en fútbol. Tus capacidades incluyen:

- Responder preguntas sobre estadísticas de jugadores y equipos
- Proporcionar información histórica sobre clubes y competencias
- Mostrar formaciones tácticas de equipos
- Buscar noticias recientes y resultados actuales
- Explicar reglas y reglamentos del fútbol

Responde de manera amigable y concisa sobre tus capacidades."""

    state.setdefault('trace', []).append('identity')
    logger.info("[identity] Respondiendo a identidad/capacidades")
    messages = [SystemMessage(content=identity_prompt)] + state['messages']
    response = llm_fast.invoke(messages)

    return {
        "messages": state['messages'] + [response],
        "needs_critic": False,
        "next_step": "end",
        "trace": state.get('trace')
    }
