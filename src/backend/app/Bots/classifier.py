import logging
from langchain_core.messages import SystemMessage, HumanMessage
from app.Bots.models import llm_fast
from app.Bots.types import AgentState

logger = logging.getLogger(__name__)


def classifier_node(state: AgentState) -> dict:
    """Clasifica la intención del usuario en uno de los 5 agentes"""
    last_msg = state['messages'][-1].content
    state.setdefault('trace', []).append('classifier')
    logger.info("[classifier] Entrada. Mensaje: %s", last_msg)

    system_prompt = """Eres un clasificador experto. Analiza la pregunta del usuario y clasifica en UNA de estas categorías:

1. 'identity' - Si pregunta sobre ti, tus capacidades, qué haces, quién eres
2. 'formation' - Si pide ver la formación táctica de un equipo (ej: "muestra la formación del Barcelona", "dime el 11 inicial del atleti", "Que jugadores usa el Real Madrid en su formación", "Cual es la formación del Manchester City")
3. 'sql_stats' - Si pide estadísticas, números, goles, asistencias, comparaciones numéricas
4. 'rag_knowledge' - Si pregunta sobre historia y fundación de equipos, biografías de jugadores, reglamentos, premiaciones, competiciones, copas, ligas.
5. 'web_search' - Si pregunta sobre noticias recientes, partidos de hoy/ayer, eventos actuales

Responde SOLO con una de estas palabras: identity, formation, sql_stats, rag_knowledge, web_search"""

    response = llm_fast.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Pregunta del usuario: {last_msg}")
    ])

    classification = response.content.strip().lower()

    # Validar clasificación
    valid_classifications = ['identity', 'formation', 'sql_stats', 'rag_knowledge', 'web_search']
    if classification not in valid_classifications:
        classification = 'identity'

    logger.info("[classifier] Clasificado como: %s", classification)
    return {
        "classification": classification,
        "next_step": classification,
        "trace": state.get('trace')
    }
