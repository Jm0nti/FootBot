import logging
from langchain_core.messages import HumanMessage, AIMessage
from app.Bots.models import llm_fast
from app.Bots.tools import formation_image_tool
from app.Bots.types import AgentState

logger = logging.getLogger(__name__)


def formation_node(state: AgentState) -> dict:
    """Maneja solicitudes de formaciones tácticas"""
    last_msg = state['messages'][-1].content
    state.setdefault('trace', []).append('formation')
    extraction_prompt = f"""Extrae SOLO el nombre del equipo de esta pregunta. \nPregunta: {last_msg}\nResponde SOLO con el nombre del equipo, nada más."""
    team_extraction = llm_fast.invoke([HumanMessage(content=extraction_prompt)])
    team_name = team_extraction.content.strip()
    logger.info("[formation] Equipo extraído: %s", team_name)

    try:
        formation_result = formation_image_tool.invoke({"team_name": team_name})
    except Exception as e:
        logger.exception("[formation] Error invocando formation_image_tool: %s", e)
        formation_result = {"image_url": None, "text": f"Error al obtener formación: {e}", "type": "formation"}

    response_text = formation_result['text']
    logger.info("[formation] Resultado: %s", response_text)

    return {
        "messages": state['messages'] + [AIMessage(content=response_text)],
        "formation_data": formation_result,
        "needs_critic": False,
        "next_step": "end",
        "trace": state.get('trace')
    }
