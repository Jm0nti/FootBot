import logging
from langchain_core.messages import HumanMessage, AIMessage
from app.Bots.models import llm_fast
from app.Bots.types import AgentState

logger = logging.getLogger(__name__)


def critic_node(state: AgentState) -> dict:
    if not state.get('needs_critic', False):
        return {"next_step": "end"}

    last_message = state['messages'][-1].content
    original_question = state['messages'][0].content
    origin = state.get('origin')
    trace = state.get('trace', [])
    web_attempts = state.get('web_search_attempts', 0)
    
    critic_prompt = f"""Eres un agente que descarta respuestas vacías, debes rechazar respuestas vacias, acepta cualquier tipo de respuesta.

Pregunta original: {original_question}
Respuesta del agente: {last_message}

responde aprobadas con: APPROVED
responde rechazadas con: REJECTED"""

    state.setdefault('trace', []).append('critic')
    logger.info("[critic] Evaluando respuesta. Pregunta: %s, \n Respuesta: %s \n Origin: %s, Web attempts: %d", 
                original_question, last_message, origin, web_attempts)
    
    evaluation = llm_fast.invoke([HumanMessage(content=critic_prompt)])
    eval_text = evaluation.content.strip()
    logger.info("[critic] Resultado de evaluación: %s", eval_text)

    if "REJECTED" in eval_text.upper():
        # CASO 1: Respuesta rechazada de RAG/SQL y NO hemos intentado web_search
        if origin in ("rag_agent", "sql_agent") and web_attempts == 0:
            logger.info("[critic] Rechazado de %s, redirigiendo a web_search (intento 1)", origin)
            return {
                "messages": [HumanMessage(content=original_question)],
                "next_step": "web_search",
                "trace": trace,
                "web_search_attempts": 1,
                "origin": origin,
                "needs_critic": False,  # Resetear para que web_search lo active
            }
        
        # CASO 2: Respuesta rechazada de web_search o ya intentamos web_search
        # En este caso, terminamos definitivamente
        logger.info("[critic] Rechazado de %s después de %d intentos web. Finalizando.", 
                    origin, web_attempts)
        rejection_msg = AIMessage(
            content="Lo siento, no pude generar una respuesta coherente para tu consulta. ¿Podrías reformular tu pregunta o ser más específico?"
        )
        return {
            "messages": state['messages'][:-1] + [rejection_msg],
            "next_step": "end",
            "trace": trace,
        }

    # CASO 3: Respuesta APROBADA
    logger.info("[critic] Respuesta APROBADA de %s", origin)
    return {
        "next_step": "end", 
        "trace": trace
    }