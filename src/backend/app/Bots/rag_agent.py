import logging
from langchain_core.messages import AIMessage, SystemMessage
from app.Bots.models import llm_smart
from app.Bots.tools import faiss_retriever
from app.Bots.types import AgentState

logger = logging.getLogger(__name__)


def rag_agent_node(state: AgentState) -> dict:
    """Agente que usa RAG para responder sobre historia y conocimiento general"""
    state.setdefault('trace', []).append('rag_agent')
    logger.info("[rag_agent] Ejecutando RAG con mensaje: %s", state['messages'][-1].content)
    system_prompt = """Eres un experto en historia del fútbol, biografías y reglamentos.
Tienes acceso a una base de conocimiento vectorial. 

Cuando el usuario pregunte sobre historia, clubes, o reglas:
1. Usa la herramienta faiss_retriever para buscar contexto relevante
2. Basa tu respuesta en el contexto recuperado
3. Si no encuentras información, indícalo claramente"""

    messages = [SystemMessage(content=system_prompt)] + state['messages']

    llm_with_tools = llm_smart.bind_tools([faiss_retriever])
    response = llm_with_tools.invoke(messages)

    if hasattr(response, 'tool_calls') and response.tool_calls:
        messages_with_response = messages + [response]

        for tool_call in response.tool_calls:
            logger.info("[rag_agent] Ejecutando tool_call: %s", tool_call)
            try:
                tool_result = faiss_retriever.invoke(tool_call['args'])
            except Exception as e:
                logger.exception("[rag_agent] Error ejecutando faiss_retriever: %s", e)
                tool_result = f"Error recuperando contexto: {e}"
            messages_with_response.append(
                AIMessage(content=f"Contexto recuperado: {tool_result}")
            )

        final_response = llm_smart.invoke(messages_with_response)

        return {
            "messages": state['messages'] + [final_response],
            "needs_critic": True,
            "next_step": "critic",
            "origin": "rag_agent",
            "trace": state.get('trace')
        }

    return {
        "messages": state['messages'] + [response],
        "needs_critic": True,
        "next_step": "critic",
        "origin": "rag_agent",
        "trace": state.get('trace')
    }
