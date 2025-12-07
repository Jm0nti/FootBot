"""
FAISS_Creator Agent - LangGraph 1.0
==================================================================================

Agente basado en LangGraph para:
1. Resumir documentos .txt usando Groq LLM
2. Crear 3 bases de datos vectoriales FAISS categorizadas
3. Workflow con estados, nodos y flujo condicional

Arquitectura LangGraph:
- State: Manejo de estado del workflow
- Nodes: Funciones de procesamiento (summarize, vectorize, diagnostics)
- Edges: Flujo condicional basado en decisiones
- Checkpointing: Memoria de estado entre ejecuciones

SOLO REQUIERE: GROQ_API_KEY
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Literal, Tuple, TypedDict, Annotated
from datetime import datetime
import logging

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN Y CARGA DE ENVIRONMENT
# ============================================================================

current_dir = Path(__file__).resolve().parent
app_dir = current_dir.parent
env_path = app_dir / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"📁 .env encontrado en: {env_path}")
else:
    logger.error(f"❌ .env NO encontrado en: {env_path}")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(f"""
❌ ERROR: GROQ_API_KEY no encontrada.
📝 Tu .env debe contener: GROQ_API_KEY=gsk_tu_key_real_aqui
🌐 Obtén tu key en: https://console.groq.com/keys
""")

logger.info(f"✅ GROQ_API_KEY cargada")


# ============================================================================
# STATE DEFINITION - LANGGRAPH
# ============================================================================

class AgentState(TypedDict):
    """Estado del agente LangGraph"""

    # Configuración inicial
    mode: str  # "full", "summary_only", "vectordb_only", "diagnostics"
    force_summaries: bool

    # Resultados de procesamiento
    summary_results: Dict[str, Dict[str, int]]
    vectordb_results: Dict[str, Dict[str, any]]

    # Estado de ejecución
    current_step: str
    errors: List[str]
    success: bool

    # Metadata
    timestamp: str
    report_path: str


# ============================================================================
# CONFIGURACIÓN CENTRALIZADA
# ============================================================================

class Config:
    """Configuración centralizada"""

    SCRIPT_PATH = Path(__file__).resolve()
    SCRIPT_DIR = SCRIPT_PATH.parent
    APP_DIR = SCRIPT_DIR.parent
    DATA_DIR = APP_DIR / "data"

    SUMMARIES_DIR = DATA_DIR / "summaries"
    VECTOR_DBS_DIR = DATA_DIR / "vector_stores"

    PLAYERS_DIR = DATA_DIR / "biografias_jugadores"
    TEAMS_DIR = DATA_DIR / "informacion_equipos"
    RULES_DIR = DATA_DIR / "competiciones_y_reglas"

    PLAYERS_SUMMARIES = SUMMARIES_DIR / "biografias_jugadores"
    TEAMS_SUMMARIES = SUMMARIES_DIR / "informacion_equipos"
    RULES_SUMMARIES = SUMMARIES_DIR / "competiciones_y_reglas"

    PLAYERS_VECTORDB = VECTOR_DBS_DIR / "jugadores_faiss"
    TEAMS_VECTORDB = VECTOR_DBS_DIR / "equipos_faiss"
    RULES_VECTORDB = VECTOR_DBS_DIR / "reglas_faiss"

    GROQ_MODEL = "openai/gpt-oss-20b"
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_CHARS_PER_CHUNK = 90000
    CHUNK_OVERLAP_CHARS = 1000

    @classmethod
    def create_directories(cls):
        """Crea directorios necesarios"""
        dirs = [
            cls.SUMMARIES_DIR,
            cls.VECTOR_DBS_DIR,
            cls.PLAYERS_SUMMARIES,
            cls.TEAMS_SUMMARIES,
            cls.RULES_SUMMARIES,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        logger.info("✅ Estructura de directorios creada")


# ============================================================================
# TOOLS - FUNCIONALIDAD CORE
# ============================================================================

class SummarizerTool:
    """Tool de resumen de documentos"""

    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.3,
            model_name=Config.GROQ_MODEL,
            api_key=GROQ_API_KEY
        )
        logger.info(f"[SummarizerTool] Inicializado con {Config.GROQ_MODEL}")

    def _get_prompt(self, category: str) -> str:
        """Obtiene el prompt especializado según categoría"""
        prompts = {
            "biografias_jugadores": """Eres un experto en fútbol especializado en análisis de jugadores.
Crea un resumen COMPLETO incluyendo:
1. Datos básicos (nombre, nacimiento, nacionalidad)
2. Información física (altura, peso, posición, pierna)
3. Carrera profesional (equipos, fechas, logros)
4. Estadísticas (goles, asistencias, títulos)
5. Estilo de juego (fortalezas, características)
6. Reconocimientos (premios, curiosidades)

Responde SOLO con el resumen estructurado.""",

            "informacion_equipos": """Eres un experto en clubes de fútbol.
Crea un resumen COMPLETO incluyendo:
1. Identidad (nombre, apodo, fundación, ciudad)
2. Estadio (nombre, capacidad)
3. Colores y escudo
4. Historia (momentos clave, evolución)
5. Palmarés (títulos nacionales, internacionales)
6. Jugadores legendarios
7. Rivalidades
8. Datos actuales

Responde SOLO con el resumen estructurado.""",

            "competiciones_y_reglas": """Eres un experto en reglamentos y competiciones.
Crea un resumen COMPLETO incluyendo:
- Para REGLAS: descripción, casos especiales, ejemplos, cambios
- Para COMPETENCIAS: nombre, formato, historia, equipos, clasificación, premios

Responde SOLO con el resumen estructurado."""
        }
        return prompts.get(category, prompts["biografias_jugadores"])

    def summarize_document(
        self,
        file_path: Path,
        category: str,
        output_file: Path,
        force: bool = False
    ) -> Tuple[str, bool]:
        """Resume un documento o devuelve el existente"""

        # Verificar si existe y no forzar regeneración
        if output_file.exists() and not force:
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing = f.read()
                if existing.strip() and not existing.startswith("[ERROR"):
                    logger.info(f"⏭️  SALTANDO: {file_path.name}")
                    return existing, True
            except Exception as e:
                logger.warning(f"⚠️  Error leyendo resumen, regenerando: {e}")

        # Generar resumen
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                return f"[DOCUMENTO VACÍO] {file_path.name}", False

            logger.info(f"📄 Resumiendo: {file_path.name}")

            system_prompt = self._get_prompt(category)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Documento: {file_path.name}\n\n{content}")
            ]

            response = self.llm.invoke(messages)
            summary = response.content.strip()

            logger.info(f"✅ Resumen generado: {len(summary)} chars")
            return summary, False

        except Exception as e:
            logger.exception(f"❌ Error: {e}")
            return f"[ERROR] {str(e)}", False

    def summarize_category(
        self,
        category: str,
        force: bool = False
    ) -> Dict[str, int]:
        """Resume todos los documentos de una categoría"""

        source_dirs = {
            "biografias_jugadores": Config.PLAYERS_DIR,
            "informacion_equipos": Config.TEAMS_DIR,
            "competiciones_y_reglas": Config.RULES_DIR
        }

        output_dirs = {
            "biografias_jugadores": Config.PLAYERS_SUMMARIES,
            "informacion_equipos": Config.TEAMS_SUMMARIES,
            "competiciones_y_reglas": Config.RULES_SUMMARIES
        }

        source_dir = source_dirs[category]
        output_dir = output_dirs[category]
        txt_files = list(source_dir.glob("*.txt"))

        if not txt_files:
            logger.warning(f"⚠️ No hay archivos en {source_dir}")
            return {"total": 0, "processed": 0, "skipped": 0, "errors": 0}

        logger.info(f"🔄 Procesando {len(txt_files)} archivos de '{category}'")

        processed = skipped = errors = 0

        for txt_file in txt_files:
            try:
                output_file = output_dir / f"{txt_file.stem}_summary.txt"
                summary, was_skipped = self.summarize_document(
                    txt_file, category, output_file, force
                )

                if was_skipped:
                    skipped += 1
                else:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(summary)
                    processed += 1

            except Exception as e:
                logger.exception(f"❌ Error: {e}")
                errors += 1

        return {
            "total": len(txt_files),
            "processed": processed,
            "skipped": skipped,
            "errors": errors
        }


class VectorDBTool:
    """Tool de creación de Vector Databases"""

    def __init__(self):
        logger.info("[VectorDBTool] Inicializando embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        logger.info("✅ VectorDBTool inicializado")

    def _load_documents(self, summaries_dir: Path, category: str) -> List[Document]:
        """Carga documentos desde resúmenes"""
        documents = []
        txt_files = list(summaries_dir.glob("*_summary.txt"))

        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                if content.strip():
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": txt_file.name,
                            "category": category,
                            "original_file": txt_file.stem.replace("_summary", "")
                        }
                    )
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error cargando {txt_file.name}: {e}")

        return documents

    def create_vectordb(self, category: str) -> Dict[str, any]:
        """Crea VectorDB para una categoría"""
        logger.info(f"\n🧠 CREANDO VECTORDB: {category.upper()}")

        summaries_dirs = {
            "biografias_jugadores": Config.PLAYERS_SUMMARIES,
            "informacion_equipos": Config.TEAMS_SUMMARIES,
            "competiciones_y_reglas": Config.RULES_SUMMARIES
        }

        output_paths = {
            "biografias_jugadores": Config.PLAYERS_VECTORDB,
            "informacion_equipos": Config.TEAMS_VECTORDB,
            "competiciones_y_reglas": Config.RULES_VECTORDB
        }

        summaries_dir = summaries_dirs[category]
        output_path = output_paths[category]

        try:
            documents = self._load_documents(summaries_dir, category)

            if not documents:
                return {
                    "category": category,
                    "documents_loaded": 0,
                    "chunks_created": 0,
                    "success": False
                }

            logger.info(f"📊 {len(documents)} documentos cargados")

            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"✂️ {len(chunks)} chunks creados")

            vectorstore = FAISS.from_documents(chunks, self.embeddings)
            vectorstore.save_local(str(output_path))

            logger.info(f"✅ VectorDB guardada en {output_path}")

            return {
                "category": category,
                "documents_loaded": len(documents),
                "chunks_created": len(chunks),
                "output_path": str(output_path),
                "success": True
            }

        except Exception as e:
            logger.exception(f"❌ Error: {e}")
            return {
                "category": category,
                "success": False,
                "error": str(e)
            }


# ============================================================================
# LANGGRAPH NODES - FUNCIONES DE ESTADO
# ============================================================================

def initialize_node(state: AgentState) -> AgentState:
    """Nodo inicial: configuración y preparación"""
    logger.info("🚀 Inicializando FAISS Creator Agent (LangGraph)")

    Config.create_directories()

    state["timestamp"] = datetime.now().isoformat()
    state["current_step"] = "initialized"
    state["errors"] = []
    state["summary_results"] = {}
    state["vectordb_results"] = {}
    state["success"] = False

    logger.info(f"📋 Modo: {state['mode']}")
    logger.info(f"🔄 Force summaries: {state.get('force_summaries', False)}")

    return state


def summarize_node(state: AgentState) -> AgentState:
    """Nodo de resumen de documentos"""
    logger.info("\n📝 EJECUTANDO NODO: SUMMARIZE")

    state["current_step"] = "summarizing"

    try:
        summarizer = SummarizerTool()
        categories = ["biografias_jugadores", "informacion_equipos", "competiciones_y_reglas"]

        results = {}
        for category in categories:
            logger.info(f"\n{'='*60}")
            logger.info(f"CATEGORÍA: {category.upper()}")
            logger.info(f"{'='*60}")

            stats = summarizer.summarize_category(
                category,
                force=state.get("force_summaries", False)
            )
            results[category] = stats

            logger.info(
                f"✅ {category}: "
                f"{stats['processed']} generados, "
                f"{stats['skipped']} saltados, "
                f"{stats['errors']} errores"
            )

        state["summary_results"] = results

        total_errors = sum(r['errors'] for r in results.values())
        if total_errors > 0:
            state["errors"].append(f"Errores en resúmenes: {total_errors}")

        logger.info("\n✅ NODO SUMMARIZE COMPLETADO")

    except Exception as e:
        logger.exception(f"❌ Error en nodo summarize: {e}")
        state["errors"].append(f"summarize_node: {str(e)}")

    return state


def vectorize_node(state: AgentState) -> AgentState:
    """Nodo de creación de Vector Databases"""
    logger.info("\n🧠 EJECUTANDO NODO: VECTORIZE")

    state["current_step"] = "vectorizing"

    try:
        vectordb_tool = VectorDBTool()
        categories = ["biografias_jugadores", "informacion_equipos", "competiciones_y_reglas"]

        results = {}
        for category in categories:
            result = vectordb_tool.create_vectordb(category)
            results[category] = result

            if result['success']:
                logger.info(
                    f"✅ {category}: "
                    f"{result['documents_loaded']} docs, "
                    f"{result['chunks_created']} chunks"
                )
            else:
                logger.error(f"❌ {category}: FALLÓ")
                state["errors"].append(f"VectorDB {category} falló")

        state["vectordb_results"] = results

        logger.info("\n✅ NODO VECTORIZE COMPLETADO")

    except Exception as e:
        logger.exception(f"❌ Error en nodo vectorize: {e}")
        state["errors"].append(f"vectorize_node: {str(e)}")

    return state


def finalize_node(state: AgentState) -> AgentState:
    """Nodo final: guardar reporte y verificar éxito"""
    logger.info("\n📊 EJECUTANDO NODO: FINALIZE")

    state["current_step"] = "finalized"

    # Verificar éxito general
    mode = state["mode"]

    if mode in ["full", "vectordb_only"]:
        vectordb_success = all(
            r.get('success', False)
            for r in state.get("vectordb_results", {}).values()
        )
        state["success"] = vectordb_success and len(state["errors"]) == 0
    elif mode == "summary_only":
        summary_errors = sum(
            r.get('errors', 0)
            for r in state.get("summary_results", {}).values()
        )
        state["success"] = summary_errors == 0
    else:
        state["success"] = True

    # Guardar reporte
    report = {
        "timestamp": state["timestamp"],
        "mode": state["mode"],
        "force_summaries": state.get("force_summaries", False),
        "summary_results": state.get("summary_results"),
        "vectordb_results": state.get("vectordb_results"),
        "errors": state["errors"],
        "success": state["success"]
    }

    report_path = Config.VECTOR_DBS_DIR / "build_report.json"
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        state["report_path"] = str(report_path)
        logger.info(f"💾 Reporte guardado: {report_path}")
    except Exception as e:
        logger.error(f"❌ Error guardando reporte: {e}")
        state["errors"].append(f"Error guardando reporte: {str(e)}")

    # Mensaje final
    if state["success"]:
        logger.info("\n" + "✅"*35)
        logger.info("🎉 PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("✅"*35)
    else:
        logger.warning("\n" + "⚠️"*35)
        logger.warning("⚠️ PIPELINE COMPLETADO CON ERRORES")
        logger.warning("⚠️"*35)
        logger.warning(f"Errores: {state['errors']}")

    return state


def diagnostics_node(state: AgentState) -> AgentState:
    """Nodo de diagnóstico del sistema"""
    logger.info("\n🔍 EJECUTANDO NODO: DIAGNOSTICS")

    state["current_step"] = "diagnostics"

    print("\n" + "="*70)
    print("🔍 DIAGNÓSTICO DEL SISTEMA")
    print("="*70 + "\n")

    # Verificar .env
    print(f"🔍 Verificando .env:")
    print(f"   Ruta: {env_path}")
    print(f"   Existe: {'✅' if env_path.exists() else '❌'}")

    if GROQ_API_KEY:
        print(f"   ✅ GROQ_API_KEY cargada")
        print(f"   📊 Longitud: {len(GROQ_API_KEY)} caracteres")
    else:
        print(f"   ❌ GROQ_API_KEY NO encontrada")

    # Verificar estructura de datos
    print(f"\n📂 Verificando estructura de datos:")
    categories = {
        "biografias_jugadores": Config.PLAYERS_DIR,
        "informacion_equipos": Config.TEAMS_DIR,
        "competiciones_y_reglas": Config.RULES_DIR
    }

    for name, path in categories.items():
        if path.exists():
            txt_count = len(list(path.glob("*.txt")))
            print(f"   ✅ {name}: {txt_count} archivos")
        else:
            print(f"   ❌ {name}: NO EXISTE")

    # Verificar resúmenes
    print(f"\n📋 Verificando resúmenes:")
    summary_dirs = {
        "biografias_jugadores": Config.PLAYERS_SUMMARIES,
        "informacion_equipos": Config.TEAMS_SUMMARIES,
        "competiciones_y_reglas": Config.RULES_SUMMARIES
    }

    for name, path in summary_dirs.items():
        if path.exists():
            summary_count = len(list(path.glob("*_summary.txt")))
            print(f"   ✅ {name}: {summary_count} resúmenes")
        else:
            print(f"   ⚠️ {name}: NO EXISTE")

    print("\n" + "="*70)
    print("✅ Diagnóstico completado")
    print("="*70 + "\n")

    state["success"] = True
    return state


# ============================================================================
# ROUTING LOGIC - FLUJO CONDICIONAL
# ============================================================================

def route_after_init(state: AgentState) -> str:
    """Determina el siguiente nodo según el modo"""
    mode = state["mode"]

    if mode == "diagnostics":
        return "diagnostics"
    elif mode == "summary_only":
        return "summarize"
    elif mode == "vectordb_only":
        return "vectorize"
    elif mode == "full":
        return "summarize"
    else:
        return END


def route_after_summarize(state: AgentState) -> str:
    """Después de resumir, decide si vectorizar o finalizar"""
    mode = state["mode"]

    if mode == "full":
        return "vectorize"
    else:
        return "finalize"


def route_after_vectorize(state: AgentState) -> str:
    """Después de vectorizar, siempre finaliza"""
    return "finalize"


def route_after_diagnostics(state: AgentState) -> str:
    """Después de diagnóstico, termina"""
    return END


# ============================================================================
# GRAPH CONSTRUCTION - LANGGRAPH
# ============================================================================

def create_faiss_graph():
    """Construye el grafo LangGraph"""

    # Crear grafo
    workflow = StateGraph(AgentState)

    # Agregar nodos
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("vectorize", vectorize_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("diagnostics", diagnostics_node)

    # Definir punto de entrada
    workflow.set_entry_point("initialize")

    # Definir edges condicionales
    workflow.add_conditional_edges(
        "initialize",
        route_after_init,
        {
            "diagnostics": "diagnostics",
            "summarize": "summarize",
            "vectorize": "vectorize",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "summarize",
        route_after_summarize,
        {
            "vectorize": "vectorize",
            "finalize": "finalize"
        }
    )

    workflow.add_conditional_edges(
        "vectorize",
        route_after_vectorize,
        {
            "finalize": "finalize"
        }
    )

    workflow.add_conditional_edges(
        "diagnostics",
        route_after_diagnostics,
        {
            END: END
        }
    )

    workflow.add_edge("finalize", END)

    # Compilar con memoria
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app


# ============================================================================
# AGENT CLASS - INTERFAZ PRINCIPAL
# ============================================================================

class FAISSCreatorAgent:
    """Agente LangGraph para creación de FAISS VectorDBs"""

    def __init__(self):
        logger.info("🤖 Inicializando FAISS Creator Agent (LangGraph)")
        self.graph = create_faiss_graph()

    def run_full_pipeline(self, force_summaries: bool = False) -> Dict:
        """Ejecuta pipeline completo"""
        logger.info("\n🚀 EJECUTANDO PIPELINE COMPLETO")

        initial_state = {
            "mode": "full",
            "force_summaries": force_summaries
        }

        config = {"configurable": {"thread_id": "faiss_full"}}

        final_state = self.graph.invoke(initial_state, config)

        return final_state

    def run_summary_only(self, force: bool = False) -> Dict:
        """Ejecuta solo resúmenes"""
        logger.info("\n📝 EJECUTANDO SOLO RESÚMENES")

        initial_state = {
            "mode": "summary_only",
            "force_summaries": force
        }

        config = {"configurable": {"thread_id": "faiss_summary"}}

        final_state = self.graph.invoke(initial_state, config)

        return final_state

    def run_vectordb_only(self) -> Dict:
        """Ejecuta solo VectorDBs"""
        logger.info("\n🧠 EJECUTANDO SOLO VECTORDBS")

        initial_state = {
            "mode": "vectordb_only",
            "force_summaries": False
        }

        config = {"configurable": {"thread_id": "faiss_vectordb"}}

        final_state = self.graph.invoke(initial_state, config)

        return final_state

    def run_diagnostics(self) -> Dict:
        """Ejecuta diagnóstico"""
        logger.info("\n🔍 EJECUTANDO DIAGNÓSTICO")

        initial_state = {
            "mode": "diagnostics",
            "force_summaries": False
        }

        config = {"configurable": {"thread_id": "faiss_diagnostics"}}

        final_state = self.graph.invoke(initial_state, config)

        return final_state


# ============================================================================
# MAIN - INTERFAZ CLI
# ============================================================================

def main():
    """Punto de entrada principal"""

    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║           FAISS CREATOR AGENT - LangGraph 1.0                  ║
    ║           Arquitectura basada en State Graph                   ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    print("\nOpciones:")
    print("1. 🔥 Pipeline completo (resúmenes + vector DBs)")
    print("2. 📝 Solo resumir documentos")
    print("3. 🔄 Regenerar TODOS los resúmenes (force)")
    print("4. 🧠 Solo crear vector databases")
    print("5. 🔍 Diagnóstico de configuración")
    print("6. ❌ Salir")

    choice = input("\nSelecciona una opción (1-6): ").strip()

    if choice == "6":
        print("👋 Saliendo...")
        return

    agent = FAISSCreatorAgent()

    if choice == "1":
        agent.run_full_pipeline(force_summaries=False)
    elif choice == "2":
        agent.run_summary_only(force=False)
    elif choice == "3":
        confirm = input("⚠️  ¿Regenerar TODOS los resúmenes? (s/n): ").lower()
    elif choice == "4":
        agent.run_vectordb_only()

main()
