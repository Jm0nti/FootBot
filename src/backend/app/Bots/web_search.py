import logging
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.messages import HumanMessage, AIMessage
from app.Bots.models import llm_smart
from app.Bots.tools import web_search_tool
from app.Bots.types import AgentState

logger = logging.getLogger(__name__)


def web_search_node(state: AgentState) -> dict:
    state.setdefault('trace', []).append('web_search')
    logger.info("[web_search] Ejecutando web search agent. Mensaje: %s", state['messages'][-1].content)
    system_prompt = """Eres un experto en noticias y eventos actuales de fútbol.
Tienes acceso a búsqueda web en tiempo real para información reciente.

INSTRUCCIONES:
1. Cuando el usuario pregunte sobre eventos recientes, partidos, noticias o información actualizada:
   - USA la herramienta web_search_tool para buscar
   - Construye queries de búsqueda específicas y relevantes
   
2. Al recibir resultados de búsqueda:
   - Resume la información de manera clara y concisa
   - Menciona las fuentes principales
   - Indica que la información es reciente/actualizada
   
3. Sé natural y conversacional en tus respuestas
4. Si no encuentras información, admítelo honestamente
"""

    messages = [SystemMessage(content=system_prompt)] + state['messages']
    llm_with_tools = llm_smart.bind_tools([web_search_tool])
    response = llm_with_tools.invoke(messages)

    if hasattr(response, 'tool_calls') and response.tool_calls:
        messages_with_response = messages + [response]

        for tool_call in response.tool_calls:
            logger.info("[web_search] Ejecutando tool_call: %s", tool_call)
            try:
                tool_result = web_search_tool.invoke(tool_call['args'])
                messages_with_response.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call['id'])
                )
            except Exception as e:
                logger.exception("[web_search] Error al ejecutar web_search_tool: %s", e)
                messages_with_response.append(
                    ToolMessage(content=f"Error al ejecutar búsqueda: {str(e)}", tool_call_id=tool_call['id'])
                )

        final_response = llm_smart.invoke(messages_with_response)
        return {
            "messages": state['messages'] + [final_response],
            "needs_critic": True,
            "next_step": "critic",
            "trace": state.get('trace'),
            "origin": "web_search",
            "web_search_attempts": state.get('web_search_attempts', 0)
        }

    return {
        "messages": state['messages'] + [response],
        "needs_critic": True,
        "next_step": "critic",
        "trace": state.get('trace'),
        "origin": "web_search",
        "web_search_attempts": state.get('web_search_attempts', 0)
    }
