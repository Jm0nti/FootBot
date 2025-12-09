import importlib
import uuid
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from app.Bots.types import AgentState

logger = logging.getLogger(__name__)


def build_graph():
    workflow = StateGraph(AgentState)

    # Importar dinámicamente los nodos (evita import circulares)
    node_modules = [
        "classifier",
        "identity",
        "formation",
        "sql_agent",
        "rag_agent",
        "web_search",
        "critic",
    ]

    for name in node_modules:
        mod = importlib.import_module(f"app.Bots.{name}")
        node_fn = getattr(mod, f"{name}_node")
        workflow.add_node(name, node_fn)

    # Entry point
    workflow.set_entry_point("classifier")

    # Edges desde classifier (se basa en el campo next_step)
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

    # Edges a END o critic
    workflow.add_edge("identity", END)
    workflow.add_edge("formation", END)
    workflow.add_edge("sql_agent", "critic")
    workflow.add_edge("rag_agent", "critic")
    workflow.add_edge("web_search", "critic")
    
    # CAMBIO CRÍTICO: Edge condicional desde critic
    # No puede ser incondicional a web_search, debe decidir basado en next_step
    def route_from_critic(state: AgentState) -> str:
        next_step = state.get('next_step', 'end')
        logger.info("[route_from_critic] Routing to: %s", next_step)
        return next_step
    
    workflow.add_conditional_edges(
        "critic",
        route_from_critic,
        {
            "web_search": "web_search",
            "end": END
        }
    )

    return workflow.compile(checkpointer=MemorySaver())


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
            final_state = self.graph.invoke(
                {
                    "messages": [HumanMessage(content=message)],
                    "next_step": "",
                    "classification": "",
                    "needs_critic": False,
                    "formation_data": None,
                    "trace": [],
                    "web_search_attempts": 0,  # Inicializar explícitamente
                    "origin": None
                },
                config=config,
            )

            last_message = final_state['messages'][-1]
            response_text = last_message.content

            image_data = None
            if final_state.get('formation_data'):
                image_data = final_state['formation_data'].get('image_url')

            self._interaction_count += 1
            trace = final_state.get('trace') if isinstance(final_state, dict) else None
            logger.info("[SoccerBot.ask] Respuesta generada. agent=%s, trace=%s", 
                       final_state.get('classification', 'unknown'), trace)

            return {
                "answer": response_text,
                "image": image_data,
                "agent_used": final_state.get('classification', 'unknown'),
                "trace": trace,
            }

        except Exception as e:
            logger.exception("[SoccerBot.ask] Error procesando solicitud: %s", e)
            return {
                "answer": f"Error al procesar la solicitud: {str(e)}",
                "image": None,
                "agent_used": "error",
                "trace": [],
            }

    def clear_memory(self):
        """Reinicia la conversación"""
        self.thread_id = str(uuid.uuid4())
        self._interaction_count = 0