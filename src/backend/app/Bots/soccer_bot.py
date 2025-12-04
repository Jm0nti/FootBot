import os
import uuid
import sqlite3
from pathlib import Path
from typing import TypedDict, List, Union, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
import requests
from bs4 import BeautifulSoup

# --- 1. CONFIGURACIÓN DE MODELOS ---
llm_fast = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
llm_smart = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)

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
        # Ruta a la base de datos (ajustar según tu estructura)
        db_path = "data/soccer_stats.db"
        
        # Si no existe la DB, crear una demo
        if not os.path.exists(db_path):
            os.makedirs("data", exist_ok=True)
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            # Crear tabla de ejemplo
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_stats (
                    player_name TEXT,
                    season TEXT,
                    goals INTEGER,
                    assists INTEGER,
                    team TEXT
                )
            """)
            # Datos de ejemplo
            sample_data = [
                ("Lionel Messi", "2020-2021", 30, 12, "Barcelona"),
                ("Cristiano Ronaldo", "2020-2021", 29, 3, "Juventus"),
                ("Erling Haaland", "2021-2022", 42, 8, "Man City"),
            ]
            cursor.executemany("INSERT INTO player_stats VALUES (?, ?, ?, ?, ?)", sample_data)
            conn.commit()
            conn.close()
        
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
            return "No se encontraron resultados para esta consulta."
        
        # Formatear resultados
        df = pd.DataFrame(results, columns=columns)
        return f"Resultados de la consulta:\n{df.to_string(index=False)}"
    
    except Exception as e:
        return f"Error al ejecutar la consulta SQL: {str(e)}"


@tool
def faiss_retriever(query: str) -> str:
    """
    Busca información en la base de conocimiento vectorial (FAISS) sobre equipos,
    clubes, competencias, historia del fútbol y reglamentos.
    Args:
        query: Pregunta o término de búsqueda
    Returns:
        Contexto relevante recuperado de los documentos
    """
    try:
        vector_store_path = "data/faiss_index"
        
        # Si no existe, crear un índice demo
        if not os.path.exists(vector_store_path):
            os.makedirs("data", exist_ok=True)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            
            # Documentos de ejemplo
            sample_docs = [
                "El FC Barcelona fue fundado en 1899 por Joan Gamper. Es uno de los clubes más exitosos de Europa.",
                "Real Madrid ha ganado 14 títulos de la UEFA Champions League, más que cualquier otro club.",
                "La regla del fuera de juego establece que un jugador está en posición adelantada si está más cerca de la línea de meta rival que el balón y el penúltimo adversario.",
                "La Copa Mundial de la FIFA se celebra cada 4 años desde 1930, con la excepción de 1942 y 1946 debido a la Segunda Guerra Mundial.",
                "Lionel Messi ha ganado 8 Balones de Oro, más que cualquier otro jugador en la historia.",
            ]
            
            from langchain.schema import Document
            docs = [Document(page_content=text) for text in sample_docs]
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(vector_store_path)
        
        # Cargar y buscar
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vectorstore = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        return f"Contexto relevante encontrado:\n{context}"
    
    except Exception as e:
        return f"Error al buscar en la base de conocimiento: {str(e)}"


@tool
def web_search_tool(query: str) -> str:
    """
    Realiza una búsqueda en internet para obtener información actualizada sobre
    partidos recientes, noticias de fútbol y eventos actuales.
    Args:
        query: Término de búsqueda
    Returns:
        Resumen de la información encontrada
    """
    try:
        # Simulación de búsqueda web (en producción usar API real como SerpAPI, Tavily, etc.)
        search_query = query.replace(" ", "+")
        
        # Ejemplo con búsqueda simple (reemplazar con API real)
        # Por ahora devolvemos respuesta simulada
        simulated_results = f"""
        Resultados de búsqueda para '{query}':
        
        1. Información reciente sobre el tema solicitado.
        2. El partido más reciente terminó con un marcador ajustado.
        3. Las últimas noticias indican cambios en las formaciones titulares.
        
        Nota: Para resultados en tiempo real, integrar API de búsqueda como SerpAPI o Tavily.
        """
        
        return simulated_results
    
    except Exception as e:
        return f"Error en la búsqueda web: {str(e)}"


@tool
def formation_image_tool(team_name: str) -> dict:
    """
    Obtiene la imagen de formación táctica de un equipo específico.
    Args:
        team_name: Nombre del equipo
    Returns:
        Diccionario con la ruta de la imagen y texto descriptivo
    """
    try:
        formations_dir = Path("assets/formations")
        
        # Buscar archivo de formación
        team_clean = team_name.strip().replace(" ", "_")
        possible_files = [
            f"{team_clean}_Formation.png",
            f"{team_clean}.png",
            f"{team_clean}_formation.png"
        ]
        
        for filename in possible_files:
            file_path = formations_dir / filename
            if file_path.exists():
                return {
                    "image_url": f"/assets/formations/{filename}",
                    "text": f"Formación táctica del {team_name}",
                    "type": "formation"
                }
        
        # Si no existe la imagen
        return {
            "image_url": None,
            "text": f"No se encontró la formación para {team_name}. Asegúrate de que existe el archivo en assets/formations/",
            "type": "formation"
        }
    
    except Exception as e:
        return {
            "image_url": None,
            "text": f"Error al buscar formación: {str(e)}",
            "type": "formation"
        }


# --- 3. ESTADO DEL GRAFO ---
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    next_step: str
    classification: str
    needs_critic: bool
    formation_data: dict | None


# --- 4. NODOS DEL GRAFO ---

def classifier_node(state: AgentState) -> dict:
    """Clasifica la intención del usuario en uno de los 5 agentes"""
    last_msg = state['messages'][-1].content
    
    system_prompt = """Eres un clasificador experto. Analiza la pregunta del usuario y clasifica en UNA de estas categorías:

1. 'identity' - Si pregunta sobre ti, tus capacidades, qué haces, quién eres
2. 'formation' - Si pide ver la formación táctica de un equipo (ej: "muestra la formación del Barcelona, cuál es el 11 titular del Real Madrid, qué formación usa el Manchester City")
3. 'sql_stats' - Si pide estadísticas, números, goles, asistencias, comparaciones numéricas
4. 'rag_knowledge' - Si pregunta sobre historia, biografías, reglamentos, fundación de clubes
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
    
    return {
        "classification": classification,
        "next_step": classification
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
    
    messages = [SystemMessage(content=identity_prompt)] + state['messages']
    response = llm_fast.invoke(messages)
    
    return {
        "messages": state['messages'] + [response],
        "needs_critic": False,
        "next_step": "end"
    }


def formation_node(state: AgentState) -> dict:
    """Maneja solicitudes de formaciones tácticas"""
    last_msg = state['messages'][-1].content
    
    # Extraer nombre del equipo
    extraction_prompt = f"""Extrae SOLO el nombre del equipo de esta pregunta. 
Pregunta: {last_msg}
Responde SOLO con el nombre del equipo, nada más."""
    
    team_extraction = llm_fast.invoke([HumanMessage(content=extraction_prompt)])
    team_name = team_extraction.content.strip()
    
    # Obtener imagen de formación
    formation_result = formation_image_tool.invoke({"team_name": team_name})
    
    response_text = formation_result['text']
    
    return {
        "messages": state['messages'] + [AIMessage(content=response_text)],
        "formation_data": formation_result,
        "needs_critic": False,
        "next_step": "end"
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
    
    messages = [SystemMessage(content=system_prompt)] + state['messages']
    
    # Vincular herramienta SQL
    llm_with_tools = llm_smart.bind_tools([sql_executor])
    response = llm_with_tools.invoke(messages)
    
    # Si hay tool calls, ejecutarlos
    if hasattr(response, 'tool_calls') and response.tool_calls:
        messages_with_response = messages + [response]
        
        for tool_call in response.tool_calls:
            tool_result = sql_executor.invoke(tool_call['args'])
            messages_with_response.append(
                AIMessage(content=f"Resultado de SQL: {tool_result}")
            )
        
        # Generar respuesta final
        final_response = llm_smart.invoke(messages_with_response)
        
        return {
            "messages": state['messages'] + [final_response],
            "needs_critic": True,
            "next_step": "critic"
        }
    
    return {
        "messages": state['messages'] + [response],
        "needs_critic": True,
        "next_step": "critic"
    }


def rag_agent_node(state: AgentState) -> dict:
    """Agente que usa RAG para responder sobre historia y conocimiento general"""
    system_prompt = """Eres un experto en historia del fútbol, biografías y reglamentos.
Tienes acceso a una base de conocimiento vectorial. 

Cuando el usuario pregunte sobre historia, clubes, o reglas:
1. Usa la herramienta faiss_retriever para buscar contexto relevante
2. Basa tu respuesta en el contexto recuperado
3. Si no encuentras información, indícalo claramente"""
    
    messages = [SystemMessage(content=system_prompt)] + state['messages']
    
    llm_with_tools = llm_smart.bind_tools([faiss_retriever])
    response = llm_with_tools.invoke(messages)
    
    # Si hay tool calls, ejecutarlos
    if hasattr(response, 'tool_calls') and response.tool_calls:
        messages_with_response = messages + [response]
        
        for tool_call in response.tool_calls:
            tool_result = faiss_retriever.invoke(tool_call['args'])
            messages_with_response.append(
                AIMessage(content=f"Contexto recuperado: {tool_result}")
            )
        
        final_response = llm_smart.invoke(messages_with_response)
        
        return {
            "messages": state['messages'] + [final_response],
            "needs_critic": True,
            "next_step": "critic"
        }
    
    return {
        "messages": state['messages'] + [response],
        "needs_critic": True,
        "next_step": "critic"
    }


def web_search_node(state: AgentState) -> dict:
    """Agente que busca información actual en la web"""
    system_prompt = """Eres un experto en noticias de fútbol actuales.
Tienes acceso a búsqueda web para información reciente.

Cuando el usuario pregunte sobre eventos recientes:
1. Usa la herramienta web_search_tool para buscar información
2. Resume los hallazgos de manera clara
3. Indica que la información es reciente"""
    
    messages = [SystemMessage(content=system_prompt)] + state['messages']
    
    llm_with_tools = llm_smart.bind_tools([web_search_tool])
    response = llm_with_tools.invoke(messages)
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        messages_with_response = messages + [response]
        
        for tool_call in response.tool_calls:
            tool_result = web_search_tool.invoke(tool_call['args'])
            messages_with_response.append(
                AIMessage(content=f"Resultados de búsqueda: {tool_result}")
            )
        
        final_response = llm_smart.invoke(messages_with_response)
        
        return {
            "messages": state['messages'] + [final_response],
            "needs_critic": True,
            "next_step": "critic"
        }
    
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
    
    critic_prompt = f"""Eres un crítico experto. Evalúa si esta respuesta es coherente y útil.

Pregunta original: {original_question}
Respuesta del agente: {last_message}

Criterios:
1. ¿Responde directamente la pregunta?
2. ¿Es coherente y tiene sentido?
3. ¿Contiene información relevante?

Si cumple los criterios, responde: APPROVED
Si no cumple, responde: REJECTED - [breve razón]"""
    
    evaluation = llm_fast.invoke([HumanMessage(content=critic_prompt)])
    eval_text = evaluation.content.strip()
    
    if "REJECTED" in eval_text.upper():
        # Respuesta rechazada
        rejection_msg = AIMessage(
            content="Lo siento, no pude generar una respuesta coherente para tu consulta. ¿Podrías reformular tu pregunta o ser más específico?"
        )
        return {
            "messages": state['messages'][:-1] + [rejection_msg],
            "next_step": "end"
        }
    
    # Respuesta aprobada
    return {"next_step": "end"}


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
            # Ejecutar el grafo
            final_state = self.graph.invoke(
                {
                    "messages": [HumanMessage(content=message)],
                    "next_step": "",
                    "classification": "",
                    "needs_critic": False,
                    "formation_data": None
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
            
            return {
                "answer": response_text,
                "image": image_data,
                "agent_used": final_state.get('classification', 'unknown')
            }
        
        except Exception as e:
            return {
                "answer": f"Error al procesar la solicitud: {str(e)}",
                "image": None,
                "agent_used": "error"
            }
    
    def clear_memory(self):
        """Reinicia la conversación"""
        self.thread_id = str(uuid.uuid4())
        self._interaction_count = 0