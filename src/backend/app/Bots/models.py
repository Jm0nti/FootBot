import logging
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

logger = logging.getLogger(__name__)


# Stubs seguros para evitar fallos en tiempo de import cuando los paquetes externos
# no están instalados en el entorno de desarrollo. Estos stubs implementan los
# métodos mínimos usados por los módulos (invoke, bind_tools) para permitir
# ejecución y pruebas básicas sin LLMs reales.
class _DummyLLM:
    def __init__(self, name="dummy"):
        self.name = name

    def invoke(self, messages):
        try:
            from langchain_core.messages import AIMessage
        except Exception:
            # Fallback to a simple object with 'content'
            class _Simple:
                def __init__(self, content):
                    self.content = content

            content = "[stub] " + (str(messages[-1].content) if isinstance(messages, list) and messages else str(messages))
            return _Simple(content)

        content = "[stub] " + (messages[-1].content if isinstance(messages, list) and messages and hasattr(messages[-1], 'content') else str(messages))
        return AIMessage(content=content)

    def bind_tools(self, tools):
        return self


# Intentar importar y crear los LLMs reales; si fallan, usar stubs
try:
    from langchain_groq import ChatGroq  # type: ignore
except Exception:
    ChatGroq = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
except Exception:
    ChatGoogleGenerativeAI = None


if ChatGroq is not None:
    try:
        llm_fast = ChatGroq(temperature=0, model_name="openai/gpt-oss-20b")
    except Exception:
        llm_fast = _DummyLLM(name="llm_fast")
        logger.exception("No se pudo inicializar llm_fast, usando stub")
else:
    llm_fast = _DummyLLM(name="llm_fast")


if ChatGoogleGenerativeAI is not None:
    try:
        llm_smart = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)
    except Exception:
        llm_smart = _DummyLLM(name="llm_smart")
        logger.exception("No se pudo inicializar llm_smart, usando stub")
else:
    llm_smart = _DummyLLM(name="llm_smart")
