import logging
from google import genai
from google.genai import types
from langchain_core.messages import AIMessage
from app.Bots.types import AgentState

logger = logging.getLogger(__name__)

# Inicializar cliente de Gemini (asegúrate de tener GOOGLE_API_KEY en env)
gemini_client = genai.Client()


def web_search_node(state: AgentState) -> dict:
    """
    Agente de búsqueda web usando Google Grounding con Gemini.
    No requiere tools separadas, Gemini busca directamente en Google.
    """
    state.setdefault('trace', []).append('web_search')
    logger.info("[web_search] Ejecutando web search con Google Grounding. Mensaje: %s", 
                state['messages'][-1].content)
    
    try:
        # Obtener el último mensaje del usuario
        last_message = state['messages'][-1].content
        
        # Construir el contexto de conversación para Gemini
        conversation_context = _build_conversation_context(state['messages'])
        
        # Configurar la herramienta de Google Search
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        
        # Configuración para generación de contenido con grounding
        config = types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=0.2,  # Baja temperatura para respuestas más precisas
            top_p=0.9,
            response_modalities=["TEXT"],
        )
        
        # Construir el prompt optimizado
        prompt = f"""Eres un experto asistente de fútbol con acceso a búsqueda web en tiempo real.

Contexto de conversación:
{conversation_context}

Pregunta actual del usuario: {last_message}

INSTRUCCIONES:
1. Busca información actualizada y precisa sobre la pregunta
2. Proporciona respuestas específicas con datos concretos (fechas, resultados, equipos, etc.)
3. Sé conciso pero informativo
4. Si mencionas información de fuentes, hazlo de manera natural
5. Responde en español de forma conversacional

Responde la pregunta del usuario basándote en la información más reciente disponible."""

        # Realizar la búsqueda y generación con Gemini
        logger.info("[web_search] Realizando búsqueda con Google Grounding...")
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=prompt,
            config=config,
        )
        
        # Extraer información de la respuesta
        response_text = response.text
        grounding_metadata = response.candidates[0].grounding_metadata if response.candidates else None
        
        # Construir respuesta enriquecida con metadata
        enhanced_response = _build_enhanced_response(response_text, grounding_metadata)
        
        logger.info("[web_search] Búsqueda completada exitosamente")
        logger.debug("[web_search] Metadata de grounding: %s", grounding_metadata)
        
        # Crear mensaje AI con la respuesta
        ai_message = AIMessage(content=enhanced_response)
        
        return {
            "messages": state['messages'] + [ai_message],
            "needs_critic": True,
            "next_step": "critic",
            "trace": state.get('trace'),
            "origin": "web_search",
            "web_search_attempts": state.get('web_search_attempts', 0),
            "grounding_metadata": grounding_metadata  # Guardar metadata para análisis
        }
    
    except Exception as e:
        logger.exception("[web_search] Error al realizar búsqueda con Google Grounding: %s", e)
        
        # Respuesta de fallback
        error_message = AIMessage(
            content=f"Lo siento, tuve un problema al buscar información actualizada sobre tu consulta. "
                   f"Error: {str(e)[:100]}"
        )
        
        return {
            "messages": state['messages'] + [error_message],
            "needs_critic": True,
            "next_step": "critic",
            "trace": state.get('trace'),
            "origin": "web_search",
            "web_search_attempts": state.get('web_search_attempts', 0) + 1
        }


def _build_conversation_context(messages: list, max_messages: int = 3) -> str:
    """
    Construye un resumen del contexto de conversación reciente.
    
    Args:
        messages: Lista de mensajes de la conversación
        max_messages: Número máximo de mensajes a incluir
    
    Returns:
        String con el contexto de conversación
    """
    # Obtener los últimos N mensajes (excluyendo el último que ya se procesa)
    recent_messages = messages[-(max_messages + 1):-1] if len(messages) > 1 else []
    
    if not recent_messages:
        return "Esta es la primera pregunta del usuario."
    
    context_parts = []
    for msg in recent_messages:
        role = "Usuario" if hasattr(msg, 'type') and msg.type == "human" else "Asistente"
        content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
        context_parts.append(f"{role}: {content}")
    
    return "\n".join(context_parts)


def _build_enhanced_response(response_text: str, grounding_metadata) -> str:
    """
    Construye una respuesta enriquecida con información de grounding.
    
    Args:
        response_text: Texto de respuesta de Gemini
        grounding_metadata: Metadata de grounding de Google
    
    Returns:
        Respuesta enriquecida con fuentes (si están disponibles)
    """
    enhanced = response_text
    
    # Si hay metadata de grounding, agregar referencias a las fuentes
    if grounding_metadata and hasattr(grounding_metadata, 'grounding_chunks'):
        chunks = grounding_metadata.grounding_chunks
        
        if chunks and len(chunks) > 0:
            # Extraer fuentes únicas
            sources = []
            seen_domains = set()
            
            for chunk in chunks[:5]:  # Máximo 5 fuentes
                if hasattr(chunk, 'web') and chunk.web:
                    domain = chunk.web.title if hasattr(chunk.web, 'title') else chunk.web.uri
                    if domain and domain not in seen_domains:
                        sources.append(f"• {domain}")
                        seen_domains.add(domain)
            
            # Agregar fuentes al final de la respuesta si existen
            if sources:
                enhanced += "\n\n📚 *Fuentes consultadas:*\n" + "\n".join(sources)
    
    # Agregar indicador de búsqueda web realizada
    if grounding_metadata and hasattr(grounding_metadata, 'web_search_queries'):
        queries = grounding_metadata.web_search_queries
        if queries:
            logger.debug("[web_search] Queries de búsqueda utilizadas: %s", queries)
    
    return enhanced


def _format_grounding_support(grounding_metadata) -> str:
    """
    Formatea información detallada de grounding support para debugging.
    
    Args:
        grounding_metadata: Metadata de grounding
    
    Returns:
        String formateado con información de soporte
    """
    if not grounding_metadata:
        return "Sin metadata de grounding"
    
    info = []
    
    if hasattr(grounding_metadata, 'web_search_queries'):
        info.append(f"Queries: {grounding_metadata.web_search_queries}")
    
    if hasattr(grounding_metadata, 'grounding_chunks'):
        info.append(f"Chunks: {len(grounding_metadata.grounding_chunks)}")
    
    if hasattr(grounding_metadata, 'grounding_supports'):
        info.append(f"Supports: {len(grounding_metadata.grounding_supports)}")
    
    return " | ".join(info)