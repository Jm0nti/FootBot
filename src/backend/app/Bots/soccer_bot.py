import os
import uuid
from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

# --- 1. CONFIGURACIÓN DE MODELOS ---

llm_fast = ChatGroq(temperature=0, model_name="openai/gpt-oss-20b")
llm_smart = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# --- 2. DEFINICIÓN DE HERRAMIENTAS 
@tool
def sql_stats_tool(query: str):
    """Ejecuta consultas SQL sobre CSVs de Kaggle para estadísticas precisas."""
    # Lógica real de conexión a DB aquí
    return "Data simulada: Messi marcó 30 goles en la temporada 2020-2021."

@tool
def knowledge_base_retriever(query: str):
    """Busca en documentos PDF/TXT (FAISS) historias, biografías y reglamentos."""
    # Lógica real de FAISS aquí
    return "Contexto recuperado: El club Barcelona fue fundado en 1899 por Hans Gamper."

@tool
def web_search_news(query: str):
    """Busca resultados recientes o noticias en la web."""
    return "Noticia reciente: El equipo local ganó 2-1 ayer."

@tool
def pitch_formation(team_name: str):
    """Devuelve la imagen de la formación táctica."""
    # Retorna un diccionario especial que el server entenderá
    return {"image_url": f"/assets/formations/{team_name}.png", "text": f"Formación del {team_name}"}

@tool
def market_value_comparison(player_a: str, player_b: str):
    """Compara valores de mercado."""
    return f"{player_a} (80M) vs {player_b} (75M)"

tools_structured = [sql_stats_tool, market_value_comparison]
tools_unstructured = [knowledge_base_retriever, web_search_news, pitch_formation]

# --- 3. ESTADO Y NODOS ---
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    next_step: str

def classifier_node(state: AgentState):
    last_msg = state['messages'][-1].content
    system_prompt = "Clasifica en: 'structured' (datos/stats), 'unstructured' (historia/bio/reglas) o 'general'."
    response = llm_fast.invoke([SystemMessage(content=system_prompt), HumanMessage(content=last_msg)])
    intent = response.content.strip().lower()
    
    if "structured" in intent: return {"next_step": "sql_agent"}
    if "unstructured" in intent: return {"next_step": "rag_agent"}
    return {"next_step": "general"}

def sql_agent_node(state: AgentState):
    return {"messages": [llm_smart.bind_tools(tools_structured).invoke(state['messages'])]}

def rag_agent_node(state: AgentState):
    return {"messages": [llm_smart.bind_tools(tools_unstructured).invoke(state['messages'])]}

def general_node(state: AgentState):
    # Responde directamente cosas simples
    return {"messages": [llm_fast.invoke(state['messages'])]}

def critic_node(state: AgentState):
    last_message = state['messages'][-1]
    # Si hay tool calls, saltamos validación para dejar que se ejecuten
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return {"next_step": "continue"}
    
    # Lógica simple de crítico: aprobar todo por ahora para no bloquear el flujo
    return {"next_step": "end"}

# --- 4. CONSTRUCCIÓN DEL GRAFO ---
workflow = StateGraph(AgentState)
workflow.add_node("classifier", classifier_node)
workflow.add_node("sql_agent", sql_agent_node)
workflow.add_node("rag_agent", rag_agent_node)
workflow.add_node("general_agent", general_node)
workflow.add_node("critic", critic_node)
workflow.add_node("tools_structured", ToolNode(tools_structured))
workflow.add_node("tools_unstructured", ToolNode(tools_unstructured))

workflow.set_entry_point("classifier")

workflow.add_conditional_edges(
    "classifier",
    lambda x: x['next_step'],
    {"sql_agent": "sql_agent", "rag_agent": "rag_agent", "general": "general_agent"}
)

# Lógica de herramientas y crítico
def check_tools_sql(state):
    last = state['messages'][-1]
    return "tools_structured" if last.tool_calls else "critic"

def check_tools_rag(state):
    last = state['messages'][-1]
    return "tools_unstructured" if last.tool_calls else "critic"

workflow.add_conditional_edges("sql_agent", check_tools_sql, ["tools_structured", "critic"])
workflow.add_conditional_edges("rag_agent", check_tools_rag, ["tools_unstructured", "critic"])
workflow.add_edge("tools_structured", "sql_agent")
workflow.add_edge("tools_unstructured", "rag_agent")
workflow.add_edge("general_agent", END)
workflow.add_edge("critic", END)

# Compilamos con memoria para persistencia
graph = workflow.compile(checkpointer=MemorySaver())

# --- 5. LA CLASE WRAPPER (Lo que usará tu Server) ---
class SoccerBot:
    def __init__(self):
        self.thread_id = str(uuid.uuid4()) # ID único para la sesión actual
        self.graph = graph
        self._interaction_count = 0
        self._interaction_limit = 50 # Opcional

    def ask(self, message: str) -> dict:
        """
        Método principal que conecta el String del server con el Grafo.
        """
        config = {"configurable": {"thread_id": self.thread_id}}
        
        # Ejecutamos el grafo
        final_state = self.graph.invoke(
            {"messages": [HumanMessage(content=message)]}, 
            config=config
        )
        
        # Extraemos la última respuesta de la IA
        last_message = final_state['messages'][-1]
        response_text = last_message.content
        
        self._interaction_count += 1
        
        # Estructura de respuesta compatible con tu frontend
        return {
            "answer": response_text,
            # Aquí podrías detectar si la respuesta contiene info de imagen
            "image": None, 
            "agent_used": "SoccerGraph"
        }

    def clear_memory(self):
        """Reinicia la conversación generando un nuevo thread_id"""
        self.thread_id = str(uuid.uuid4())
        self._interaction_count = 0