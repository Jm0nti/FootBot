import os
import uuid
import sqlite3
from pathlib import Path
from typing import TypedDict, List, Union, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import pandas as pd
import requests
from dotenv import load_dotenv
from tavily import TavilyClient
import base64

# Cargar variables de entorno
load_dotenv()

# Logger local
import logging
logger = logging.getLogger(__name__)

# --- 1. CONFIGURACIÓN DE MODELOS ---
llm_fast = ChatGroq(temperature=0, model_name="openai/gpt-oss-20b")
llm_smart = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)

# --- 2. DEFINICIÓN DE HERRAMIENTAS ---

@tool
def sql_executor(query: str) -> str:
    """
    Ejecuta consultas SQL sobre una base de datos SQLite con estadísticas de fútbol.
    Args:
        query: Consulta SQL a ejecutar (SELECT statements)
    Returns:
        Resultados de la consulta en formato texto
    """
    try:
        logger.info("[sql_executor] Ejecutando consulta SQL: %s", query)
        # Ruta a la base de datos (ajustar según tu estructura)
        db_path = "data/soccer_stats.db"
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Validar que sea SELECT
        if not query.strip().upper().startswith("SELECT"):
            return "Error: Solo se permiten consultas SELECT por seguridad."
        
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        
        if not results:
            logger.info("[sql_executor] No se encontraron resultados para la consulta")
            return "No se encontraron resultados para esta consulta."
        
        # Formatear resultados
        df = pd.DataFrame(results, columns=columns)
        logger.info("[sql_executor] Resultado rows=%d cols=%d", len(df), len(df.columns))
        return f"Resultados de la consulta:\n{df.to_string(index=False)}"
    
    except Exception as e:
        logger.exception("[sql_executor] Error al ejecutar la consulta SQL: %s", e)
        return f"Error al ejecutar la consulta SQL: {str(e)}"


@tool
def faiss_retriever(query: str) -> str:
    """
    Busca información en la base de conocimiento vectorial (FAISS) sobre equipos,
    clubes, competencias, historia del fútbol y reglamentos.
    Usa OpenAI text-embedding-3-small para generar embeddings de alta calidad.
    Args:
        query: Pregunta o término de búsqueda
    Returns:
        Contexto relevante recuperado de los documentos
    """
    try:
        logger.info("[faiss_retriever] Buscando en FAISS: %s", query)
        # Validar que existe la API key de OpenAI
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("[faiss_retriever] OPENAI_API_KEY no configurada")
            return "Error: OPENAI_API_KEY no está configurada en el archivo .env"
        
        vector_store_path = "data/faiss_index"
        
        # Inicializar embeddings de OpenAI con text-embedding-3-small
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_api_key
        )
        
        # Cargar el índice existente
        vectorstore = FAISS.load_local(
            vector_store_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Realizar búsqueda de similitud (k=5 para obtener los 5 documentos más relevantes)
        docs = vectorstore.similarity_search(query, k=5)
        
        if not docs:
            logger.info("[faiss_retriever] No se encontraron docs para la query")
            return "No se encontró información relevante en la base de conocimiento."
        
        # Formatear el contexto recuperado
        context = "\n\n".join([f"- {doc.page_content}" for doc in docs])
        logger.info("[faiss_retriever] Documentos recuperados: %d", len(docs))
        
        return f"Contexto relevante encontrado:\n{context}"
    
    except Exception as e:
        logger.exception("[faiss_retriever] Error al buscar en la base de conocimiento: %s", e)
        return f"Error al buscar en la base de conocimiento: {str(e)}"



PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Endpoint de Perplexity
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"


# ============================================================================
# TOOL: WEB SEARCH CON PERPLEXITY
# ============================================================================

@tool
def web_search_tool(query: str) -> str:
    """
    Realiza una búsqueda en internet sobre fútbol usando Perplexity AI para 
    obtener información actualizada, precisa y con fuentes verificadas sobre 
    partidos, noticias, resultados y eventos recientes.
    
    Args:
        query: Término de búsqueda relacionado con fútbol (equipos, partidos, noticias)
    
    Returns:
        Respuesta detallada con información actualizada y fuentes citadas
    
    Ejemplos de uso:
        - "¿Cuándo juega el Real Madrid próximamente?"
        - "Últimas noticias del FC Barcelona"
        - "Resultados de LaLiga de hoy"
    """
    try:
        # Construir el prompt optimizado para búsquedas de fútbol
        search_prompt = f"""Busca información actualizada sobre: {query}

Proporciona:
1. Información específica y verificada
2. Fechas y detalles concretos si están disponibles
3. Las fuentes de donde obtuviste la información

Mantén la respuesta concisa pero informativa."""

        # Headers para la API de Perplexity
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Payload para la API
        # Usando modelo sonar-pro para mejor precisión en búsquedas
        payload = {
            "model": "sonar-pro",  # Mejor modelo para búsquedas web
            "messages": [
                {
                    "role": "system",
                    "content": "Eres un asistente experto en búsquedas de información deportiva, especialmente fútbol. Proporciona información precisa, actualizada y con fuentes verificables."
                },
                {
                    "role": "user",
                    "content": search_prompt
                }
            ],
            "temperature": 0.2,  # Baja temperatura para respuestas más precisas
            "top_p": 0.9,
            "return_citations": True,  # Importante: incluir citas
            "search_recency_filter": "month",  # Priorizar resultados del último mes
            "stream": False
        }
        
        # Realizar request a Perplexity API
        response = requests.post(
            PERPLEXITY_API_URL,
            headers=headers,
            json=payload,
            timeout=30  # Timeout de 30 segundos
        )
        
        # Verificar si la request fue exitosa
        response.raise_for_status()
        
        # Parsear respuesta
        data = response.json()
        
        # Extraer contenido y citas
        content = data['choices'][0]['message']['content']
        citations = data.get('citations', [])
        
        # Construir respuesta estructurada
        search_summary = f"🔍 **Búsqueda: '{query}'**\n\n"
        search_summary += f"{content}\n\n"
        
        # Agregar citas si existen
        if citations:
            search_summary += "📚 **Fuentes consultadas:**\n"
            for idx, citation in enumerate(citations[:5], 1):  # Máximo 5 fuentes
                search_summary += f"{idx}. {citation}\n"
        
        return search_summary
    
    except requests.exceptions.HTTPError as e:
        # Error HTTP específico
        status_code = e.response.status_code
        if status_code == 401:
            return "❌ Error de autenticación: Verifica tu API key de Perplexity."
        elif status_code == 429:
            return "❌ Límite de rate exceeded. Espera un momento e intenta de nuevo."
        else:
            return f"❌ Error HTTP {status_code}: {str(e)}"
    
    except requests.exceptions.Timeout:
        return "❌ Timeout: La búsqueda tardó demasiado. Intenta con una query más específica."
    
    except requests.exceptions.RequestException as e:
        return f"❌ Error de conexión: {str(e)}\nVerifica tu conexión a internet."
    
    except Exception as e:
        return f"❌ Error inesperado al realizar búsqueda: {str(e)}"



@tool
def formation_image_tool(team_name: str) -> dict:
    """
    Obtiene la imagen de formación táctica de un equipo específico.
    
    Args:
        team_name: Nombre del equipo (ej: "Barcelona", "Real Madrid")
    
    Returns:
        Diccionario con:
        - image_url: Ruta de la imagen (para web)
        - image_base64: Imagen codificada en base64 (para envío)
        - text: Texto descriptivo
        - type: "formation"
        - team_name: Nombre del equipo
    """
    try:
        logger.info("[formation_image_tool] Buscando formación para: %s", team_name)
        formations_dir = Path("assets/formations")
        
        # Normalizar nombre del equipo
        team_clean = team_name.strip().lower().replace(" ", "_")
        
        # Buscar archivo de formación (múltiples variantes)
        possible_files = [
            f"{team_clean}_formation.png",
            f"{team_clean}_Formation.png",
            f"{team_clean}.png",
            f"{team_name.strip().replace(' ', '_')}_formation.png",
            f"{team_name.strip().replace(' ', '_')}.png"
        ]
        
        found_file = None
        for filename in possible_files:
            file_path = formations_dir / filename
            if file_path.exists():
                found_file = file_path
                logger.info("[formation_image_tool] Imagen encontrada: %s", filename)
                break
        
        # Si se encontró la imagen
        if found_file:
            # Leer imagen y convertir a base64
            with open(found_file, 'rb') as img_file:
                image_data = img_file.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            return {
                "image_url": f"/assets/formations/{found_file.name}",
                "image_base64": image_base64,
                "text": f"📋 Formación táctica del {team_name}",
                "type": "formation",
                "team_name": team_name,
                "success": True
            }
        
        # Si no existe la imagen
        logger.warning("[formation_image_tool] No se encontró imagen para: %s", team_name)
        logger.warning("[formation_image_tool] Archivos buscados: %s", possible_files)
        
        return {
            "image_url": None,
            "image_base64": None,
            "text": f"⚠️ No se encontró la formación táctica para {team_name}. Verifica que exista el archivo en assets/formations/",
            "type": "formation",
            "team_name": team_name,
            "success": False
        }
    
    except Exception as e:
        logger.exception("[formation_image_tool] Error procesando formación: %s", e)
        return {
            "image_url": None,
            "image_base64": None,
            "text": f"❌ Error al obtener la formación: {str(e)}",
            "type": "formation",
            "team_name": team_name,
            "success": False,
            "error": str(e)
        }



# --- 3. ESTADO DEL GRAFO ---
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    next_step: str
    classification: str
    needs_critic: bool
    formation_data: dict | None
    trace: List[str]


# --- 4. NODOS DEL GRAFO ---

def classifier_node(state: AgentState) -> dict:
    """Clasifica la intención del usuario en uno de los 5 agentes"""
    last_msg = state['messages'][-1].content
    state.setdefault('trace', []).append('classifier')
    logger.info("[classifier] Entrada. Mensaje: %s", last_msg)

    system_prompt = """Eres un clasificador experto. Analiza la pregunta del usuario y clasifica en UNA de estas categorías:

1. 'identity' - Si pregunta sobre ti, tus capacidades, qué haces, quién eres, o un saludo general
2. 'formation' - Si pide ver la formación táctica de un equipo (ej: "muestra la formación del Barcelona, cual es la formación del Real Madrid, dame el 11 titular del liverpool")
3. 'sql_stats' - Si pide estadísticas, números, goles, asistencias, comparaciones numéricas
4. 'rag_knowledge' - Si pregunta sobre historia, biografías, reglamentos, fundación de clubes
5. 'web_search' - Si pregunta sobre noticias recientes, partidos de hoy/ayer, eventos actuales, o preguntas que se salen de tu base de conocimiento que estén el el dominio del fútbol

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


def formation_node(state: dict[str, any]) -> dict:
    """
    Maneja solicitudes de formaciones tácticas.
    
    Flujo:
    1. Extrae el nombre del equipo de la pregunta del usuario
    2. Busca la imagen de formación usando formation_image_tool
    3. Prepara la respuesta con la imagen (si existe)
    4. Retorna el estado actualizado
    
    Args:
        state: Estado del agente (AgentState) con:
            - messages: Lista de mensajes
            - trace: Lista de pasos ejecutados
            - formation_data: Datos de la formación (se agregará)
    
    Returns:
        Estado actualizado con:
        - messages: Mensajes + respuesta del agente
        - formation_data: Datos de la imagen de formación
        - needs_critic: False (no necesita validación)
        - next_step: "end" (termina el flujo)
        - trace: Traza actualizada
    """
    last_msg = state['messages'][-1].content
    logger.info("[formation_node] Procesando solicitud de formación")
    
    # Inicializar trace si no existe
    state.setdefault('trace', []).append('formation')
    
    # ========== PASO 1: EXTRAER NOMBRE DEL EQUIPO ==========
    extraction_prompt = f"""Analiza esta pregunta y extrae SOLAMENTE el nombre del equipo.

Pregunta: {last_msg}

INSTRUCCIONES:
- Devuelve SOLO el nombre del equipo, sin explicaciones
- Usa el nombre oficial común (ej: "Barcelona" no "FC Barcelona")
- Si hay múltiples equipos, devuelve el primero mencionado
- Si no hay equipo claro, devuelve "No especificado"

Ejemplos:
- "Muéstrame la alineación del Barcelona" → Barcelona
- "Formación del Real Madrid" → Real Madrid
- "¿Cuál es el 11 inicial del PSG?" → PSG

Equipo:"""
    
    try:
        team_extraction = llm_fast.invoke([HumanMessage(content=extraction_prompt)])
        team_name = team_extraction.content.strip()
        logger.info("[formation_node] Equipo extraído: '%s'", team_name)
        
        # Validar que se extrajo un equipo
        if not team_name or team_name.lower() in ["no especificado", "ninguno", "no hay"]:
            logger.warning("[formation_node] No se pudo extraer un equipo válido")
            return {
                "messages": state['messages'] + [AIMessage(
                    content="⚠️ No pude identificar el equipo del que quieres ver la formación. Por favor, especifica el nombre del equipo."
                )],
                "formation_data": None,
                "needs_critic": False,
                "next_step": "end",
                "trace": state.get('trace')
            }
    
    except Exception as e:
        logger.exception("[formation_node] Error extrayendo nombre del equipo: %s", e)
        return {
            "messages": state['messages'] + [AIMessage(
                content=f"❌ Error al procesar tu solicitud: {str(e)}"
            )],
            "formation_data": None,
            "needs_critic": False,
            "next_step": "end",
            "trace": state.get('trace')
        }
    
    # ========== PASO 2: OBTENER IMAGEN DE FORMACIÓN ==========
    try:
        logger.info("[formation_node] Invocando formation_image_tool para: %s", team_name)
        formation_result = formation_image_tool.invoke({"team_name": team_name})
        
    except Exception as e:
        logger.exception("[formation_node] Error invocando formation_image_tool: %s", e)
        formation_result = {
            "image_url": None,
            "image_base64": None,
            "text": f"❌ Error al obtener la formación: {str(e)}",
            "type": "formation",
            "team_name": team_name,
            "success": False
        }
    
    # ========== PASO 3: PREPARAR RESPUESTA ==========
    response_text = formation_result.get('text', '')
    success = formation_result.get('success', False)
    
    logger.info("[formation_node] Resultado - Success: %s, Text: %s", success, response_text)
    
    # Si hay imagen, agregar información adicional
    if success and formation_result.get('image_url'):
        response_text += "\n\n💡 Puedes ver la formación táctica en la imagen mostrada arriba."
    
    # ========== PASO 4: RETORNAR ESTADO ACTUALIZADO ==========
    return {
        "messages": state['messages'] + [AIMessage(content=response_text)],
        "formation_data": formation_result,  # IMPORTANTE: Esto se usa en el servidor
        "needs_critic": False,
        "next_step": "end",
        "trace": state.get('trace')
    }


def sql_agent_node(state: AgentState) -> dict:
    """Agente que maneja consultas de estadísticas con SQL"""
    system_prompt = """Eres un experto en estadísticas de fútbol. Tienes acceso a una base de datos SQL.

Tablas disponibles:
- player_stats (player_name, season, goals, assists, team)

Cuando el usuario pida estadísticas:
1. Genera una consulta SQL SELECT apropiada
2. Usa la herramienta sql_executor para ejecutarla
3. Interpreta los resultados y responde al usuario

Si no puedes responder con SQL, dilo claramente."""
    
    state.setdefault('trace', []).append('sql_agent')
    logger.info("[sql_agent] Iniciando SQL agent con mensaje: %s", state['messages'][-1].content)
    messages = [SystemMessage(content=system_prompt)] + state['messages']
    
    # Vincular herramienta SQL
    llm_with_tools = llm_smart.bind_tools([sql_executor])
    response = llm_with_tools.invoke(messages)
    
    # Si hay tool calls, ejecutarlos
    if hasattr(response, 'tool_calls') and response.tool_calls:
        messages_with_response = messages + [response]
        
        for tool_call in response.tool_calls:
            logger.info("[sql_agent] Ejecutando tool_call: %s", tool_call)
            try:
                tool_result = sql_executor.invoke(tool_call['args'])
            except Exception as e:
                logger.exception("[sql_agent] Error ejecutando sql_executor: %s", e)
                tool_result = f"Error ejecutando SQL: {e}"
            messages_with_response.append(
                AIMessage(content=f"Resultado de SQL: {tool_result}")
            )
        
        # Generar respuesta final
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


def rag_agent_node(state: AgentState) -> dict:
    """Agente que usa RAG para responder sobre historia y conocimiento general"""
    system_prompt = """Eres un experto en historia del fútbol, biografías y reglamentos.
Tienes acceso a una base de conocimiento vectorial. 

Cuando el usuario pregunte sobre historia, clubes, o reglas:
1. Usa la herramienta faiss_retriever para buscar contexto relevante
2. Basa tu respuesta en el contexto recuperado
3. Si no encuentras información, indícalo claramente"""
    
    state.setdefault('trace', []).append('rag_agent')
    logger.info("[rag_agent] Ejecutando RAG con mensaje: %s", state['messages'][-1].content)
    messages = [SystemMessage(content=system_prompt)] + state['messages']
    
    llm_with_tools = llm_smart.bind_tools([faiss_retriever])
    response = llm_with_tools.invoke(messages)
    
    # Si hay tool calls, ejecutarlos
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
            "trace": state.get('trace')
        }
    
    return {
        "messages": state['messages'] + [response],
        "needs_critic": True,
        "next_step": "critic",
        "trace": state.get('trace')
    }





def web_search_node(state: dict) -> dict:
    """
    Nodo del agente que busca información actual en la web sobre fútbol usando Perplexity.
    
    Este agente:
    1. Recibe la pregunta del usuario
    2. Decide si necesita usar web_search_tool
    3. Ejecuta la búsqueda si es necesario
    4. Genera una respuesta natural basada en los resultados
    5. Pasa al nodo crítico para validación
    
    Args:
        state: Estado del agente con estructura AgentState
            - messages: Lista de mensajes de la conversación
            - needs_critic: Bool indicando si pasa por validación
            - next_step: Siguiente nodo en el grafo
    
    Returns:
        Estado actualizado con respuesta del agente
    """
    
    # System prompt que define el comportamiento del agente
    system_prompt = """Eres un experto en noticias y eventos actuales de fútbol.
Tienes acceso a búsqueda web en tiempo real usando Perplexity AI para información precisa y actualizada.

INSTRUCCIONES:
1. Cuando el usuario pregunte sobre eventos recientes, partidos, noticias o información actualizada:
   - USA la herramienta web_search_tool para buscar
   - Construye queries de búsqueda claras y específicas en español
   - Incluye contexto relevante en la query (fechas, competiciones, etc.)
   
2. Al recibir resultados de búsqueda:
   - Resume la información de manera clara y concisa
   - Destaca los datos más importantes (fechas, resultados, nombres)
   - Menciona que la información proviene de fuentes actualizadas
   
3. Sé natural y conversacional en tus respuestas
4. Si no encuentras información específica, sugiere alternativas o admítelo honestamente

EJEMPLOS DE QUERIES EFECTIVAS:
- Usuario: "¿Cuándo juega el Real Madrid?" 
  → Query: "Real Madrid próximo partido fecha horario 2024"
  
- Usuario: "Últimas noticias del Barcelona"
  → Query: "FC Barcelona noticias últimas fichajes resultados"
  
- Usuario: "¿Quién ganó ayer en LaLiga?"
  → Query: "LaLiga resultados partido ayer marcador"

IMPORTANTE: Perplexity proporciona información muy precisa, confía en sus resultados.
"""
    
    # Construir mensajes para el LLM
    messages = [SystemMessage(content=system_prompt)] + state['messages']
    
    # Vincular la tool al LLM (permite que el LLM decida cuándo usarla)
    llm_with_tools = llm_smart.bind_tools([web_search_tool])
    
    # Primera invocación: LLM decide si usar la tool
    response = llm_with_tools.invoke(messages)
    
    # CASO 1: El LLM decidió usar la tool
    if hasattr(response, 'tool_calls') and response.tool_calls:
        # Agregar la respuesta del LLM con tool_calls al historial
        messages_with_response = messages + [response]
        
        # Ejecutar cada tool call solicitada
        for tool_call in response.tool_calls:
            try:
                # Invocar la tool con los argumentos que el LLM proporcionó
                tool_result = web_search_tool.invoke(tool_call['args'])
                
                # CRÍTICO: Usar ToolMessage con el tool_call_id correcto
                # Esto permite al LLM asociar el resultado con la llamada
                messages_with_response.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call['id']  # ID que vincula call con resultado
                    )
                )
            except Exception as e:
                # Si falla la tool, informar al LLM del error
                messages_with_response.append(
                    ToolMessage(
                        content=f"Error al ejecutar búsqueda: {str(e)}",
                        tool_call_id=tool_call['id']
                    )
                )
        
        # Segunda invocación: LLM genera respuesta final con los resultados
        final_response = llm_smart.invoke(messages_with_response)
        
        return {
            "messages": state['messages'] + [final_response],
            "needs_critic": True,  # Pasa por validación del crítico
            "next_step": "critic"
        }
    
    # CASO 2: El LLM no usó la tool (responde directamente)
    # Esto puede pasar si la pregunta no requiere búsqueda web
    return {
        "messages": state['messages'] + [response],
        "needs_critic": True,
        "next_step": "critic"
    }






def critic_node(state: AgentState) -> dict:
    """Verifica la calidad y coherencia de las respuestas"""
    if not state.get('needs_critic', False):
        return {"next_step": "end"}
    
    last_message = state['messages'][-1].content
    original_question = state['messages'][0].content
    
    critic_prompt = f"""Aprueba todas las respuestas que te lleguen, esto es para debugging.

Pregunta original: {original_question}
Respuesta del agente: {last_message}

Criterios:
1. ¿Responde directamente la pregunta?
2. ¿Es coherente y tiene sentido?
3. ¿Contiene información relevante?

Si cumple los criterios, responde: APPROVED
Si no cumple, responde: REJECTED - [breve razón]"""
    
    state.setdefault('trace', []).append('critic')
    logger.info("[critic] Evaluando respuesta. Pregunta: %s", original_question)
    evaluation = llm_fast.invoke([HumanMessage(content=critic_prompt)])
    eval_text = evaluation.content.strip()
    logger.info("[critic] Resultado de evaluación: %s", eval_text)
    
    if "REJECTED" in eval_text.upper():
        # Respuesta rechazada
        rejection_msg = AIMessage(
            content="Lo siento, no pude verificar que la respuesta generada por los agentes fuera adecuada. Por favor, intenta reformular tu pregunta o proporcionar más detalles."
        )
        return {
            "messages": state['messages'][:-1] + [rejection_msg],
            "next_step": "end"
        }
    
    # Respuesta aprobada
    return {"next_step": "end", "trace": state.get('trace')}


# --- 5. CONSTRUCCIÓN DEL GRAFO ---
def build_graph():
    workflow = StateGraph(AgentState)
    
    # Agregar nodos
    workflow.add_node("classifier", classifier_node)
    workflow.add_node("identity", identity_node)
    workflow.add_node("formation", formation_node)
    workflow.add_node("sql_agent", sql_agent_node)
    workflow.add_node("rag_agent", rag_agent_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("critic", critic_node)
    
    # Entry point
    workflow.set_entry_point("classifier")
    
    # Edges desde classifier
    def route_from_classifier(state: AgentState) -> str:
        return state['next_step']
    
    workflow.add_conditional_edges(
        "classifier",
        route_from_classifier,
        {
            "identity": "identity",
            "formation": "formation",
            "sql_stats": "sql_agent",
            "rag_knowledge": "rag_agent",
            "web_search": "web_search"
        }
    )
    
    # Edges a END
    workflow.add_edge("identity", END)
    workflow.add_edge("formation", END)
    workflow.add_edge("sql_agent", "critic")
    workflow.add_edge("rag_agent", "critic")
    workflow.add_edge("web_search", "critic")
    workflow.add_edge("critic", END)
    
    return workflow.compile(checkpointer=MemorySaver())


# --- 6. CLASE WRAPPER ---
class SoccerBot:
    def __init__(self):
        self.thread_id = str(uuid.uuid4())
        self.graph = build_graph()
        self._interaction_count = 0
        self._interaction_limit = 50
    
    def ask(self, message: str) -> dict:
        """Método principal que procesa preguntas del usuario"""
        config = {"configurable": {"thread_id": self.thread_id}}
        
        try:
            logger.info("[SoccerBot.ask] Invocando grafo con mensaje: %s", message)
            # Ejecutar el grafo
            final_state = self.graph.invoke(
                {
                    "messages": [HumanMessage(content=message)],
                    "next_step": "",
                    "classification": "",
                    "needs_critic": False,
                    "formation_data": None,
                    "trace": []
                },
                config=config
            )
            
            # Extraer respuesta
            last_message = final_state['messages'][-1]
            response_text = last_message.content
            
            # Verificar si hay datos de formación
            image_data = None
            if final_state.get('formation_data'):
                image_data = final_state['formation_data'].get('image_url')
            
            self._interaction_count += 1
            trace = final_state.get('trace') if isinstance(final_state, dict) else None
            logger.info("[SoccerBot.ask] Respuesta generada. agent=%s, trace=%s", final_state.get('classification', 'unknown'), trace)
            
            return {
                "answer": response_text,
                "image": image_data,
                "agent_used": final_state.get('classification', 'unknown'),
                "trace": trace
            }
        
        except Exception as e:
            logger.exception("[SoccerBot.ask] Error procesando solicitud: %s", e)
            return {
                "answer": f"Error al procesar la solicitud: {str(e)}",
                "image": None,
                "agent_used": "error",
                "trace": []
            }
    
    def clear_memory(self):
        """Reinicia la conversación"""
        self.thread_id = str(uuid.uuid4())
        self._interaction_count = 0