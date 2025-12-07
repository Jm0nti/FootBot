"""
FAISS_Creator Agent - Preprocesamiento de Documentos y Creación de Vector DBs
==================================================================================

Este agente se ejecuta de forma independiente para:
1. Resumir documentos .txt usando Groq LLM (SALTA RESÚMENES EXISTENTES)
2. Crear 3 bases de datos vectoriales FAISS categorizadas usando embeddings de HuggingFace (gratuitos)
3. Preparar el conocimiento para el sistema principal

NO forma parte del flujo conversacional, solo prepara los datos.
SOLO REQUIERE: GROQ_API_KEY

v2.1 - Detecta y salta automáticamente resúmenes ya existentes
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Literal, Tuple
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings  # Embeddings GRATUITOS
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
current_dir = Path(__file__).resolve().parent  # bots/
app_dir = current_dir.parent                    # app/

# El .env está en app/ directamente
env_path = app_dir / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"📁 .env encontrado en: {env_path}")
else:
    logger.error(f"❌ .env NO encontrado en: {env_path}")

# Validar SOLO Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    error_msg = f"""
❌ ERROR: GROQ_API_KEY no encontrada.

🔧 Ubicaciones verificadas:
   - {env_path}

📝 Tu .env debe contener:
   GROQ_API_KEY=gsk_tu_key_real_aqui

🌐 Obtén tu key en: https://console.groq.com/keys
"""
    raise ValueError(error_msg)

logger.info(f"✅ GROQ_API_KEY cargada (primeros 10 chars: {GROQ_API_KEY[:10]}...)")


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

class FAISS_Creator:
    """Configuración centralizada del FAISS_Creator"""

    SCRIPT_PATH = Path(__file__).resolve()
    SCRIPT_DIR = SCRIPT_PATH.parent           # bots/
    APP_DIR = SCRIPT_DIR.parent               # app/
    DATA_DIR = APP_DIR / "data"               # app/data/

    SUMMARIES_DIR = DATA_DIR / "summaries"
    VECTOR_DBS_DIR = DATA_DIR / "vector_stores"

    # Subdirectorios de documentos originales
    PLAYERS_DIR = DATA_DIR / "biografias_jugadores"
    TEAMS_DIR = DATA_DIR / "informacion_equipos"
    RULES_DIR = DATA_DIR / "competiciones_y_reglas"

    # Subdirectorios de summaries
    PLAYERS_SUMMARIES = SUMMARIES_DIR / "biografias_jugadores"
    TEAMS_SUMMARIES = SUMMARIES_DIR / "informacion_equipos"
    RULES_SUMMARIES = SUMMARIES_DIR / "competiciones_y_reglas"

    # Vector DBs paths
    PLAYERS_VECTORDB = VECTOR_DBS_DIR / "jugadores_faiss"
    TEAMS_VECTORDB = VECTOR_DBS_DIR / "equipos_faiss"
    RULES_VECTORDB = VECTOR_DBS_DIR / "reglas_faiss"

    # Modelos

    GROQ_MODEL = "llama-3.1-8b-instant" #Para textos cortos

    # Embeddings de HuggingFace (GRATUITOS, no requieren API key)
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Chunking
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    @classmethod
    def create_directories(cls):
        """Crea solo los directorios que no existen (summaries y vector_stores)"""
        dirs = [
            cls.SUMMARIES_DIR,
            cls.VECTOR_DBS_DIR,
            cls.PLAYERS_SUMMARIES,
            cls.TEAMS_SUMMARIES,
            cls.RULES_SUMMARIES,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        logger.info("✅ Estructura de directorios de salida creada")

        # Verificar que existan los directorios de entrada
        logger.info(f"\n📂 Verificando directorios de entrada:")
        logger.info(f"   Data dir: {cls.DATA_DIR}")

        missing_dirs = []
        for name, dir_path in [
            ("biografias_jugadores", cls.PLAYERS_DIR),
            ("informacion_equipos", cls.TEAMS_DIR),
            ("competiciones_y_reglas", cls.RULES_DIR)
        ]:
            if dir_path.exists():
                txt_count = len(list(dir_path.glob("*.txt")))
                logger.info(f"   ✅ {name}: {txt_count} archivos .txt")
            else:
                logger.warning(f"   ❌ {name}: NO EXISTE - {dir_path}")
                missing_dirs.append(str(dir_path))

        if missing_dirs:
            logger.warning(f"\n⚠️ ADVERTENCIA: Faltan carpetas con datos")
            logger.warning(f"Asegúrate de que existan estas carpetas:")
            for d in missing_dirs:
                logger.warning(f"  - {d}")


# ============================================================================
# TOOL 1: DOCUMENT SUMMARIZER (CON DETECCIÓN DE RESÚMENES EXISTENTES)
# ============================================================================

class DocumentSummarizer:
    """
    Tool que resume documentos .txt usando Groq LLM.

    🆕 v2.1 Características:
    - Detecta automáticamente resúmenes existentes y los salta
    - Solo procesa documentos nuevos o faltantes
    - Modo 'force' para regenerar todos los resúmenes
    - Estadísticas detalladas: generados, saltados, errores
    """

    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.3,
            model_name=FAISS_Creator.GROQ_MODEL,
            api_key=GROQ_API_KEY
        )
        logger.info(f"[DocumentSummarizer] Inicializado con {FAISS_Creator.GROQ_MODEL}")

    def _get_summary_prompt(
        self,
        category: Literal["biografias_jugadores", "informacion_equipos", "competiciones_y_reglas"]
    ) -> str:
        """Retorna el prompt especializado según la categoría"""

        prompts = {
            "biografias_jugadores": """Eres un experto en fútbol especializado en análisis de jugadores.

Tu tarea es crear un resumen COMPLETO Y DETALLADO de la información sobre este jugador.

El resumen debe incluir:
1. **Datos básicos**: Nombre completo, apodo, fecha de nacimiento, nacionalidad
2. **Información física**: Altura, peso, posición principal, pierna dominante
3. **Carrera profesional**: Equipos donde ha jugado (con fechas), logros importantes
4. **Estadísticas destacadas**: Goles, asistencias, títulos ganados
5. **Estilo de juego**: Fortalezas, características técnicas, rol táctico
6. **Datos contextuales**: Premios individuales, reconocimientos, curiosidades

IMPORTANTE:
- Mantén TODA la información relevante del documento original
- Usa formato estructurado con secciones claras
- No inventes datos, solo resume lo que está en el documento
- Si faltan datos, omite esa sección
- El resumen puede ser extenso (500-800 palabras) para mantener detalle

Responde SOLO con el resumen estructurado, sin preámbulos.""",

            "informacion_equipos": """Eres un experto en historia y análisis de clubes de fútbol.

Tu tarea es crear un resumen COMPLETO Y DETALLADO de la información sobre este equipo.

El resumen debe incluir:
1. **Identidad del club**: Nombre completo, apodo, año de fundación, ciudad/país
2. **Estadio**: Nombre, capacidad, características
3. **Colores y escudo**: Descripción de la identidad visual
4. **Historia**: Momentos clave, eras doradas, evolución del club
5. **Palmarés**: Títulos nacionales, internacionales, otros logros
6. **Jugadores legendarios**: Ídolos históricos y actuales
7. **Rivalidades**: Clásicos y rivalidades importantes
8. **Datos actuales**: Entrenador, liga, situación reciente

IMPORTANTE:
- Mantén TODA la información relevante del documento original
- Usa formato estructurado con secciones claras
- Preserva fechas, números y datos específicos
- El resumen puede ser extenso (600-1000 palabras) para mantener contexto
- No inventes información

Responde SOLO con el resumen estructurado, sin preámbulos.""",

            "competiciones_y_reglas": """Eres un experto en reglamentos y competiciones de fútbol.

Tu tarea es crear un resumen COMPLETO Y DETALLADO de esta información sobre reglas o competencias.

El resumen debe incluir:

Para REGLAS/REGLAMENTOS:
1. **Regla o aspecto**: Qué regla o aspecto del juego se describe
2. **Descripción detallada**: Explicación completa de cómo funciona
3. **Casos especiales**: Excepciones, situaciones particulares
4. **Ejemplos**: Casos de aplicación práctica
5. **Cambios recientes**: Si aplica, cambios en el reglamento

Para COMPETENCIAS:
1. **Nombre y tipo**: Nombre oficial, categoría (liga, copa, etc.)
2. **Formato**: Sistema de competición, número de equipos
3. **Historia**: Año de fundación, datos históricos relevantes
4. **Equipos participantes**: Principales clubes o selecciones
5. **Sistema de clasificación**: Cómo se determina el ganador
6. **Premios**: Títulos, clasificaciones a otras competencias
7. **Récords y estadísticas**: Datos destacados

IMPORTANTE:
- Mantén precisión en reglas y formatos
- Preserva números, fechas y datos específicos
- Usa formato claro y estructurado
- El resumen puede ser extenso (500-900 palabras)
- No simplificar excesivamente las reglas

Responde SOLO con el resumen estructurado, sin preámbulos."""
        }

        return prompts[category]

    def summarize_document(
        self,
        file_path: Path,
        category: Literal["biografias_jugadores", "informacion_equipos", "competiciones_y_reglas"],
        output_file: Path,
        force: bool = False
    ) -> Tuple[str, bool]:
        """
        Resume un documento usando Groq LLM.

        Args:
            file_path: Ruta del archivo .txt a resumir
            category: Categoría del documento
            output_file: Ruta donde se guardará el resumen
            force: Si es True, regenera el resumen aunque ya exista

        Returns:
            Tupla (resumen, was_skipped)
            - resumen: Texto del resumen generado o existente
            - was_skipped: True si se saltó porque ya existía
        """

        # 🆕 VERIFICAR SI YA EXISTE EL RESUMEN
        if output_file.exists() and not force:
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_summary = f.read()

                if existing_summary.strip() and not existing_summary.startswith("[ERROR"):
                    logger.info(f"⏭️  SALTANDO: {file_path.name} (resumen ya existe: {len(existing_summary)} chars)")
                    return existing_summary, True
                else:
                    logger.warning(f"⚠️  Resumen existente inválido, regenerando: {file_path.name}")
            except Exception as e:
                logger.warning(f"⚠️  Error leyendo resumen existente, regenerando: {e}")

        # Si llegamos aquí, debemos generar el resumen
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                logger.warning(f"⚠️ Archivo vacío: {file_path.name}")
                return f"[DOCUMENTO VACÍO] {file_path.name}", False

            logger.info(f"📄 Resumiendo: {file_path.name} ({len(content)} chars)")

            system_prompt = self._get_summary_prompt(category)

            user_message = f"""Documento a resumir: {file_path.name}

CONTENIDO:
{content}

Genera el resumen siguiendo las instrucciones del system prompt."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]

            response = self.llm.invoke(messages)
            summary = response.content.strip()

            logger.info(f"✅ Resumen generado: {len(summary)} chars")
            return summary, False

        except Exception as e:
            logger.exception(f"❌ Error resumiendo {file_path.name}: {e}")
            return f"[ERROR AL RESUMIR] {file_path.name}: {str(e)}", False

    def summarize_category(
        self,
        category: Literal["biografias_jugadores", "informacion_equipos", "competiciones_y_reglas"],
        force: bool = False
    ) -> Dict[str, int]:
        """
        Resume todos los documentos de una categoría.

        Args:
            category: Categoría a procesar
            force: Si es True, regenera todos los resúmenes aunque ya existan

        Returns:
            Estadísticas: total, processed, skipped, errors
        """

        source_dirs = {
            "biografias_jugadores": FAISS_Creator.PLAYERS_DIR,
            "informacion_equipos": FAISS_Creator.TEAMS_DIR,
            "competiciones_y_reglas": FAISS_Creator.RULES_DIR
        }

        output_dirs = {
            "biografias_jugadores": FAISS_Creator.PLAYERS_SUMMARIES,
            "informacion_equipos": FAISS_Creator.TEAMS_SUMMARIES,
            "competiciones_y_reglas": FAISS_Creator.RULES_SUMMARIES
        }

        source_dir = source_dirs[category]
        output_dir = output_dirs[category]

        txt_files = list(source_dir.glob("*.txt"))

        if not txt_files:
            logger.warning(f"⚠️ No se encontraron archivos .txt en {source_dir}")
            return {"total": 0, "processed": 0, "skipped": 0, "errors": 0}

        logger.info(f"🔄 Procesando {len(txt_files)} archivos de categoría '{category}'")

        if force:
            logger.warning(f"🔥 MODO FORCE activado: regenerando TODOS los resúmenes")

        processed = 0
        skipped = 0
        errors = 0

        for txt_file in txt_files:
            try:
                output_file = output_dir / f"{txt_file.stem}_summary.txt"

                summary, was_skipped = self.summarize_document(
                    txt_file,
                    category,
                    output_file,
                    force=force
                )

                if was_skipped:
                    skipped += 1
                else:
                    # Solo escribimos si generamos un resumen nuevo
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(summary)
                    logger.info(f"💾 Guardado: {output_file.name}")
                    processed += 1

            except Exception as e:
                logger.exception(f"❌ Error procesando {txt_file.name}: {e}")
                errors += 1

        return {
            "total": len(txt_files),
            "processed": processed,
            "skipped": skipped,
            "errors": errors
        }

    def summarize_all(self, force: bool = False) -> Dict[str, Dict[str, int]]:
        """
        Resume todos los documentos de todas las categorías.

        Args:
            force: Si es True, regenera todos los resúmenes aunque ya existan
        """
        logger.info("🚀 Iniciando resumen de todos los documentos")

        if force:
            logger.warning("🔥 MODO FORCE: Se regenerarán TODOS los resúmenes")

        results = {}
        categories = ["biografias_jugadores", "informacion_equipos", "competiciones_y_reglas"]

        for category in categories:
            logger.info(f"\n{'='*60}")
            logger.info(f"CATEGORÍA: {category.upper()}")
            logger.info(f"{'='*60}")

            stats = self.summarize_category(category, force=force)
            results[category] = stats

            logger.info(
                f"✅ {category}: "
                f"{stats['processed']} generados, "
                f"{stats['skipped']} saltados, "
                f"{stats['errors']} errores"
            )

        # Mostrar resumen total
        total_processed = sum(r['processed'] for r in results.values())
        total_skipped = sum(r['skipped'] for r in results.values())
        total_errors = sum(r['errors'] for r in results.values())

        logger.info(f"\n{'='*60}")
        logger.info("📊 RESUMEN TOTAL")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Generados: {total_processed}")
        logger.info(f"⏭️  Saltados: {total_skipped}")
        logger.info(f"❌ Errores: {total_errors}")

        return results


# ============================================================================
# TOOL 2: FAISS VECTOR DB CREATOR (CON HUGGINGFACE EMBEDDINGS - GRATUITO)
# ============================================================================

class FAISSVectorDBCreator:
    """
    Tool que crea 3 bases de datos vectoriales FAISS separadas.

    Características:
    - Crea una FAISS DB por categoría (jugadores, equipos, reglas)
    - Usa embeddings de HuggingFace (GRATUITOS, multilingües)
    - Chunking inteligente de documentos
    - Metadata para tracking de fuente
    - NO requiere API keys adicionales
    """

    def __init__(self):
        logger.info("[FAISSVectorDBCreator] Inicializando embeddings de HuggingFace...")
        logger.info("⏳ Primera ejecución: descargando modelo (puede tardar 1-2 min)...")

        # HuggingFace Embeddings - COMPLETAMENTE GRATUITO
        # Modelo multilingüe optimizado para español
        self.embeddings = HuggingFaceEmbeddings(
            model_name=FAISS_Creator.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},  # Usa CPU (cambia a 'cuda' si tienes GPU)
            encode_kwargs={'normalize_embeddings': True}  # Mejora la similitud
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=FAISS_Creator.CHUNK_SIZE,
            chunk_overlap=FAISS_Creator.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        logger.info(f"✅ Embeddings inicializados: {FAISS_Creator.EMBEDDING_MODEL}")

    def _load_documents_from_summaries(
        self,
        summaries_dir: Path,
        category: str
    ) -> List[Document]:
        """Carga archivos de resúmenes en Documents de LangChain."""

        documents = []
        txt_files = list(summaries_dir.glob("*_summary.txt"))

        if not txt_files:
            logger.warning(f"⚠️ No se encontraron resúmenes en {summaries_dir}")
            return documents

        logger.info(f"📂 Cargando {len(txt_files)} resúmenes de {category}")

        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                if not content.strip():
                    logger.warning(f"⚠️ Resumen vacío: {txt_file.name}")
                    continue

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": txt_file.name,
                        "category": category,
                        "original_file": txt_file.stem.replace("_summary", "")
                    }
                )

                documents.append(doc)
                logger.info(f"✅ Cargado: {txt_file.name} ({len(content)} chars)")

            except Exception as e:
                logger.exception(f"❌ Error cargando {txt_file.name}: {e}")

        return documents

    def create_vectordb_for_category(
        self,
        category: Literal["biografias_jugadores", "informacion_equipos", "competiciones_y_reglas"]
    ) -> Dict[str, any]:
        """Crea una base de datos vectorial FAISS para una categoría."""

        logger.info(f"\n{'='*60}")
        logger.info(f"CREANDO VECTORDB: {category.upper()}")
        logger.info(f"{'='*60}")

        summaries_dirs = {
            "biografias_jugadores": FAISS_Creator.PLAYERS_SUMMARIES,
            "informacion_equipos": FAISS_Creator.TEAMS_SUMMARIES,
            "competiciones_y_reglas": FAISS_Creator.RULES_SUMMARIES
        }

        output_paths = {
            "biografias_jugadores": FAISS_Creator.PLAYERS_VECTORDB,
            "informacion_equipos": FAISS_Creator.TEAMS_VECTORDB,
            "competiciones_y_reglas": FAISS_Creator.RULES_VECTORDB
        }

        summaries_dir = summaries_dirs[category]
        output_path = output_paths[category]

        try:
            documents = self._load_documents_from_summaries(summaries_dir, category)

            if not documents:
                logger.warning(f"⚠️ No hay documentos para procesar en {category}")
                return {
                    "category": category,
                    "documents_loaded": 0,
                    "chunks_created": 0,
                    "success": False
                }

            logger.info(f"📊 {len(documents)} documentos cargados")

            logger.info("✂️ Aplicando chunking...")
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"✅ {len(chunks)} chunks creados")

            logger.info("🧠 Generando embeddings y creando FAISS index...")
            vectorstore = FAISS.from_documents(chunks, self.embeddings)

            logger.info(f"💾 Guardando vectorstore en {output_path}")
            vectorstore.save_local(str(output_path))

            logger.info(f"✅ VectorDB creada exitosamente para {category}")

            return {
                "category": category,
                "documents_loaded": len(documents),
                "chunks_created": len(chunks),
                "output_path": str(output_path),
                "success": True
            }

        except Exception as e:
            logger.exception(f"❌ Error creando vectorDB para {category}: {e}")
            return {
                "category": category,
                "documents_loaded": 0,
                "chunks_created": 0,
                "success": False,
                "error": str(e)
            }

    def create_all_vectordbs(self) -> Dict[str, Dict[str, any]]:
        """Crea las 3 bases de datos vectoriales."""

        logger.info("\n" + "="*70)
        logger.info("🚀 INICIANDO CREACIÓN DE BASES DE DATOS VECTORIALES")
        logger.info("="*70 + "\n")

        results = {}
        categories = ["biografias_jugadores", "informacion_equipos", "competiciones_y_reglas"]

        for category in categories:
            result = self.create_vectordb_for_category(category)
            results[category] = result

        logger.info("\n" + "="*70)
        logger.info("📊 RESUMEN FINAL")
        logger.info("="*70)

        for category, stats in results.items():
            if stats['success']:
                logger.info(f"✅ {category}: {stats['documents_loaded']} docs, {stats['chunks_created']} chunks")
            else:
                logger.info(f"❌ {category}: FALLÓ")

        return results


# ============================================================================
# FAISS_Creator AGENT (Orquestador)
# ============================================================================

class FAISS_CreatorAgent:
    """
    Agente orquestador que ejecuta el pipeline completo de construcción de conocimiento.

    Pipeline:
    1. Crear estructura de directorios
    2. Resumir todos los documentos (Tool 1 - Groq) - SALTA EXISTENTES
    3. Crear bases de datos vectoriales (Tool 2 - HuggingFace embeddings gratuitos)
    4. Generar reporte de ejecución

    SOLO REQUIERE: GROQ_API_KEY
    """

    def __init__(self):
        logger.info("🤖 Inicializando FAISS Creator Agent")
        self.summarizer = DocumentSummarizer()
        self.vectordb_creator = FAISSVectorDBCreator()
        FAISS_Creator.create_directories()

    def run_full_pipeline(self, force_summaries: bool = False) -> Dict[str, any]:
        """
        Ejecuta el pipeline completo de construcción de conocimiento.

        Args:
            force_summaries: Si es True, regenera todos los resúmenes
        """

        logger.info("\n" + "🔥"*35)
        logger.info("🚀 INICIANDO PIPELINE COMPLETO DE FAISS CREATOR")
        logger.info("🔥"*35 + "\n")

        report = {
            "timestamp": None,
            "summary_results": None,
            "vectordb_results": None,
            "success": False
        }

        try:
            logger.info("\n📝 PASO 1: RESUMIENDO DOCUMENTOS")
            logger.info("-" * 70)
            summary_results = self.summarizer.summarize_all(force=force_summaries)
            report['summary_results'] = summary_results

            logger.info("\n🧠 PASO 2: CREANDO BASES DE DATOS VECTORIALES")
            logger.info("-" * 70)
            vectordb_results = self.vectordb_creator.create_all_vectordbs()
            report['vectordb_results'] = vectordb_results

            all_success = all(
                result['success'] for result in vectordb_results.values()
            )

            report['success'] = all_success

            from datetime import datetime
            report['timestamp'] = datetime.now().isoformat()

            report_path = FAISS_Creator.VECTOR_DBS_DIR / "build_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"\n💾 Reporte guardado en: {report_path}")

            if all_success:
                logger.info("\n" + "✅"*35)
                logger.info("🎉 PIPELINE COMPLETADO EXITOSAMENTE")
                logger.info("✅"*35 + "\n")
            else:
                logger.warning("\n" + "⚠️"*35)
                logger.warning("⚠️ PIPELINE COMPLETADO CON ERRORES")
                logger.warning("⚠️"*35 + "\n")

            return report

        except Exception as e:
            logger.exception(f"❌ Error crítico en el pipeline: {e}")
            report['success'] = False
            report['error'] = str(e)
            return report

    def run_summary_only(self, force: bool = False) -> Dict[str, Dict[str, int]]:
        """
        Ejecuta solo el paso de resúmenes

        Args:
            force: Si es True, regenera todos los resúmenes aunque ya existan
        """
        logger.info("📝 Ejecutando SOLO resúmenes de documentos")
        if force:
            logger.warning("🔥 MODO FORCE: regenerando TODOS los resúmenes")
        return self.summarizer.summarize_all(force=force)

    def run_vectordb_only(self) -> Dict[str, Dict[str, any]]:
        """Ejecuta solo el paso de creación de vector DBs"""
        logger.info("🧠 Ejecutando SOLO creación de vector databases")
        return self.vectordb_creator.create_all_vectordbs()


# ============================================================================
# SCRIPT DE EJECUCIÓN
# ============================================================================

def main():
    """Punto de entrada principal"""

    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║           FAIS CREATOR AGENT v2.1                         ║
    ║           (Con detección de resúmenes existentes)              ║
    ║           Solo requiere GROQ_API_KEY                           ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    print("\nOpciones:")
    print("1. 🔥 Ejecutar pipeline completo (resúmenes + vector DBs)")
    print("2. 📝 Solo resumir documentos (salta existentes)")
    print("3. 🔄 Regenerar TODOS los resúmenes (force)")
    print("4. 🧠 Solo crear vector databases (requiere resúmenes previos)")
    print("5. 🔍 Diagnóstico de configuración")
    print("6. ❌ Salir")

    choice = input("\nSelecciona una opción (1-6): ").strip()

    if choice == "6":
        print("👋 Saliendo...")
        return

    if choice == "5":
        run_diagnostics()
        return

    agent = FAISS_CreatorAgent()

    if choice == "1":
        agent.run_full_pipeline(force_summaries=False)
    elif choice == "2":
        agent.run_summary_only(force=False)
    elif choice == "3":
        confirm = input("⚠️  ¿Seguro que quieres regenerar TODOS los resúmenes? (s/n): ").strip().lower()
        if confirm == 's':
            agent.run_summary_only(force=True)
        else:
            print("❌ Operación cancelada")
            return
    elif choice == "4":
        agent.run_vectordb_only()
    else:
        print("❌ Opción inválida")
        return

    print("\n✅ Proceso finalizado. Revisa los logs para más detalles.")


def run_diagnostics():
    """Ejecuta un diagnóstico completo del sistema"""
    print("\n" + "="*70)
    print("🔍 DIAGNÓSTICO DEL SISTEMA")
    print("="*70 + "\n")

    # 1. Verificar ubicación del script
    script_path = Path(__file__).resolve()
    print(f"📍 Ubicación del script:")
    print(f"   {script_path}")
    print(f"   Carpeta: {script_path.parent}\n")

    # 2. Verificar directorio de ejecución
    print(f"📁 Directorio de ejecución actual:")
    print(f"   {Path.cwd()}\n")

    # 3. Buscar .env en app/
    app_dir = script_path.parent.parent  # app/
    env_path = app_dir / ".env"

    print(f"🔍 Buscando .env en app/:")
    print(f"   Ruta esperada: {env_path}")

    if env_path.exists():
        print(f"   ✅ Encontrado\n")

        # Leer y mostrar contenido (sin mostrar la key completa)
        try:
            with open(env_path, 'r') as f:
                content = f.read()
                if "GROQ_API_KEY" in content:
                    print(f"   ✅ Contiene GROQ_API_KEY")

                    # Intentar cargar
                    load_dotenv(dotenv_path=env_path)
                    groq_key = os.getenv("GROQ_API_KEY")

                    if groq_key:
                        print(f"   ✅ Key cargada correctamente")
                        print(f"   📊 Longitud: {len(groq_key)} caracteres")
                        print(f"   🔒 Primeros 15 chars: {groq_key[:15]}...")
                        print(f"   🔒 Últimos 5 chars: ...{groq_key[-5:]}")
                    else:
                        print(f"   ❌ Key NO se pudo cargar")
                else:
                    print(f"   ❌ NO contiene GROQ_API_KEY")
                    print(f"   📝 Contenido actual:")
                    print(f"   {content[:200]}")
        except Exception as e:
            print(f"   ⚠️ Error leyendo archivo: {e}")
    else:
        print(f"   ❌ NO existe\n")
        print(f"💡 Crea el archivo en: {env_path}")
        print(f"📝 Con este contenido:")
        print(f"   GROQ_API_KEY=gsk_tu_key_real_aqui\n")

    # 4. Verificar estructura de data/
    print(f"\n📂 Verificando estructura de data/:")
    data_dir = app_dir / "data"  # app/data/
    print(f"   Ubicación: {data_dir}")

    if data_dir.exists():
        print(f"   ✅ data/ existe\n")

        required_dirs = [
            "biografias_jugadores",
            "informacion_equipos",
            "competiciones_y_reglas"
        ]

        for dir_name in required_dirs:
            dir_path = data_dir / dir_name
            if dir_path.exists():
                txt_files = list(dir_path.glob("*.txt"))
                txt_count = len(txt_files)
                print(f"   ✅ {dir_name}/ - {txt_count} archivos .txt")
                if txt_count > 0:
                    print(f"      Ejemplos: {', '.join([f.name for f in txt_files[:3]])}")
            else:
                print(f"   ❌ {dir_name}/ NO existe")
                print(f"      Ruta esperada: {dir_path}")
    else:
        print(f"   ❌ data/ NO existe")
        print(f"   💡 Debes crear: {data_dir}")
        print(f"   Con las subcarpetas:")
        print(f"      - biografias_jugadores/")
        print(f"      - informacion_equipos/")
        print(f"      - competiciones_y_reglas/")

    # 5. Verificar resúmenes existentes
    print(f"\n📋 Verificando resúmenes existentes:")
    summaries_dir = data_dir / "summaries"

    if summaries_dir.exists():
        print(f"   ✅ summaries/ existe")

        summary_dirs = {
            "biografias_jugadores": summaries_dir / "biografias_jugadores",
            "informacion_equipos": summaries_dir / "informacion_equipos",
            "competiciones_y_reglas": summaries_dir / "competiciones_y_reglas"
        }

        for cat_name, cat_path in summary_dirs.items():
            if cat_path.exists():
                summary_files = list(cat_path.glob("*_summary.txt"))
                print(f"   ✅ {cat_name}: {len(summary_files)} resúmenes")
            else:
                print(f"   ⚠️ {cat_name}: carpeta no existe")
    else:
        print(f"   ⚠️ summaries/ NO existe (se creará al ejecutar)")

    # 6. Test de conexión con Groq (si la key está disponible)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print(f"\n🧪 Probando conexión con Groq API...")
        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import HumanMessage

            llm = ChatGroq(
                temperature=0,
                model_name="llama-3.1-8b-instant",
                api_key=groq_key
            )

            response = llm.invoke([HumanMessage(content="Di 'OK' si funcionó")])
            print(f"   ✅ Conexión exitosa!")
            print(f"   📝 Respuesta: {response.content}")
        except Exception as e:
            print(f"   ❌ Error de conexión: {e}")

    print("\n" + "="*70)
    print("✅ Diagnóstico completado")
    print("="*70 + "\n")

    # Resumen de acciones necesarias
    print("📋 RESUMEN DE ACCIONES NECESARIAS:")

    actions_needed = []

    if not env_path.exists() or not os.getenv("GROQ_API_KEY"):
        actions_needed.append(f"1. Crear/verificar {env_path} con GROQ_API_KEY")

    if not data_dir.exists():
        actions_needed.append(f"2. Crear carpeta {data_dir}")

    for dir_name in ["biografias_jugadores", "informacion_equipos", "competiciones_y_reglas"]:
        dir_path = data_dir / dir_name
        if not dir_path.exists():
            actions_needed.append(f"3. Crear carpeta {dir_path}")

    if actions_needed:
        for action in actions_needed:
            print(f"   ⚠️ {action}")
    else:
        print(f"   ✅ Todo está configurado correctamente")

    print()


if __name__ == "__main__":
    main()