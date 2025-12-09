from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    next_step: str
    classification: str
    needs_critic: bool
    formation_data: dict | None
    trace: List[str]
    # Opcionales para control de reintentos entre agentes
    origin: str | None
    tried_web_search: bool
    web_search_attempts: int
