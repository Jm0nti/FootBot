import logging
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from app.Bots.models import llm_smart
from app.Bots.tools import faiss_retriever
from app.Bots.types import AgentState
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def rag_agent_node(state: AgentState) -> dict:
    """Agente que usa RAG segmentado para responder sobre fútbol."""
    
    # Prompt actualizado para forzar la clasificación de categoría
    system_prompt = """Eres un experto en conocimiento futbolístico con acceso a tres bases de datos vectoriales especializadas (RAG).

Tus fuentes de información se dividen en:
1. 'equipos': Historia y fundación de clubes/equipos (Real Madrid, Barcelona, etc.).
2. 'jugadores': Biografías, fecha y lugar de nacimiento, trayectoria y logros de futbolistas.
3. 'reglas': Reglamentos, faltas, arbitraje, posiciones (defensa, centrocampista, etc.), competiciones (Ligas como la Bundesliga, Copas como Copa América) y premios (Balón de Oro, Bota de Oro, etc.).

INSTRUCCIONES:
1. Analiza la pregunta del usuario e identifica a qué categoría pertenece.
2. USA la herramienta 'faiss_retriever' enviando la 'query' y la 'category' correcta ("equipos", "jugadores" o "reglas").
3. Si la pregunta abarca varios temas (ej: "¿Quién ganó el Balón de Oro jugando para el Real Madrid?"), prioriza la categoría más relevante para obtener la respuesta (en este caso 'reglas' o 'equipos' dependiendo del enfoque, pero elige solo una llamada principal o haz dos si es estrictamente necesario).
4. Basa tu respuesta EXCLUSIVAMENTE en el contexto recuperado.
5. Si no encuentras información, dilo honestamente."""
    
    state.setdefault('trace', []).append('rag_agent')
    logger.info("[rag_agent] Ejecutando RAG con mensaje: %s", state['messages'][-1].content)
    messages = [SystemMessage(content=system_prompt)] + state['messages']
    
    llm_with_tools = llm_smart.bind_tools([faiss_retriever])
    response = llm_with_tools.invoke(messages)
    
    # Si hay tool calls, ejecutarlos
    if hasattr(response, 'tool_calls') and response.tool_calls:
        messages_with_response = messages + [response]
        
        for tool_call in response.tool_calls:
            try:
                # Extraer argumentos, ahora incluye 'category'
                tool_args = tool_call['args']
                logger.info("[rag_agent] Ejecutando tool con args: %s", tool_args)
                
                tool_result = faiss_retriever.invoke(tool_args)
            except Exception as e:
                logger.exception("[rag_agent] Error en tool: %s", e)
                tool_result = f"Error recuperando contexto: {e}"
            
            messages_with_response.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call['id']
                )
            )
        
        # Generar respuesta final con el contexto obtenido
        final_response = llm_smart.invoke(messages_with_response)
        
        return {
            "messages": state['messages'] + [final_response],
            "needs_critic": True,
            "next_step": "critic",
            "trace": state.get('trace')
        }
    
    return {
        "messages": state['messages'] + [response],
        "needs_critic": True,
        "next_step": "critic",
        "trace": state.get('trace')
    }