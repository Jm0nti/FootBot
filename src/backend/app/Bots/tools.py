import os
import sqlite3
from pathlib import Path
import logging
import pandas as pd
import requests

# Importaciones opcionales: envolver en try/except para que el módulo no falle
# si las dependencias no están instaladas en el entorno de desarrollo.
try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None

try:
    from langchain_core.tools import tool
except Exception:
    # Decorador fallback que simplemente devuelve la función sin modificación
    def tool(fn=None, **kwargs):
        if fn is None:
            def _inner(f):
                return f
            return _inner
        return fn

# Nota: FAISS y OpenAIEmbeddings se importan dentro de la función faiss_retriever
# para evitar errores durante la importación del paquete cuando dichas libs
# no están instaladas.

logger = logging.getLogger(__name__)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Endpoint de Perplexity
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

perplexity_client = None
if TavilyClient is not None and PERPLEXITY_API_KEY:
    try:
        perplexity_client = TavilyClient(api_key=PERPLEXITY_API_KEY)
    except Exception:
        perplexity_client = None
        logging.getLogger(__name__).exception("No se pudo inicializar perplexity_client")


@tool
def sql_executor(query: str) -> str:
    """Ejecuta una consulta SQL `SELECT` sobre la base de datos local y devuelve resultados formateados."""
    try:
        logger.info("[sql_executor] Ejecutando consulta SQL: %s", query)
        db_path = "data/soccer_stats.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        if not query.strip().upper().startswith("SELECT"):
            return "Error: Solo se permiten consultas SELECT por seguridad."

        cursor.execute(query)
        results = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()

        if not results:
            logger.info("[sql_executor] No se encontraron resultados para la consulta")
            return "No se encontraron resultados para esta consulta."

        df = pd.DataFrame(results, columns=columns)
        logger.info("[sql_executor] Resultado rows=%d cols=%d", len(df), len(df.columns))
        return f"Resultados de la consulta:\n{df.to_string(index=False)}"

    except Exception as e:
        logger.exception("[sql_executor] Error al ejecutar la consulta SQL: %s", e)
        return f"Error al ejecutar la consulta SQL: {str(e)}"


@tool
def faiss_retriever(query: str) -> str:
    """Recupera contexto relevante desde un vectorstore FAISS para la `query` dada."""
    try:
        logger.info("[faiss_retriever] Buscando en FAISS: %s", query)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("[faiss_retriever] OPENAI_API_KEY no configurada")
            return "Error: OPENAI_API_KEY no está configurada en el archivo .env"

        # Importar FAISS y embeddings de forma perezosa
        try:
            from langchain_community.vectorstores import FAISS
        except Exception:
            logger.exception("[faiss_retriever] langchain_community.vectorstores no disponible")
            return "Error: FAISS no está instalado en este entorno."

        try:
            from langchain_openai import OpenAIEmbeddings
        except Exception:
            logger.exception("[faiss_retriever] langchain_openai.OpenAIEmbeddings no disponible")
            return "Error: OpenAIEmbeddings no están disponibles en este entorno."

        vector_store_path = "data/faiss_index"
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_api_key
        )

        vectorstore = FAISS.load_local(
            vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        docs = vectorstore.similarity_search(query, k=5)

        if not docs:
            logger.info("[faiss_retriever] No se encontraron docs para la query")
            return "No se encontró información relevante en la base de conocimiento."

        context = "\n\n".join([f"- {doc.page_content}" for doc in docs])
        logger.info("[faiss_retriever] Documentos recuperados: %d", len(docs))
        return f"Contexto relevante encontrado:\n{context}"

    except Exception as e:
        logger.exception("[faiss_retriever] Error al buscar en la base de conocimiento: %s", e)
        return f"Error al buscar en la base de conocimiento: {str(e)}"


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
    """Busca una imagen de formación para `team_name` en `assets/formations` y devuelve metadatos."""
    try:
        logger.info("[formation_image_tool] Buscando formación para: %s", team_name)
        formations_dir = Path("assets/formations")
        team_clean = team_name.strip().replace(" ", "_")
        possible_files = [
            f"{team_clean}_Formation.png",
            f"{team_clean}.png",
            f"{team_clean}_formation.png"
        ]

        for filename in possible_files:
            file_path = formations_dir / filename
            if file_path.exists():
                logger.info("[formation_image_tool] Imagen encontrada: %s", filename)
                return {
                    "image_url": f"/assets/formations/{filename}",
                    "text": f"Formación táctica del {team_name}",
                    "type": "formation"
                }

        logger.info("[formation_image_tool] No se encontró imagen para: %s", team_name)
        return {
            "image_url": None,
            "text": f"No se encontró la formación para {team_name}. Asegúrate de que existe el archivo en assets/formations/",
            "type": "formation"
        }

    except Exception as e:
        logger.exception("[formation_image_tool] Error al buscar formación: %s", e)
        return {
            "image_url": None,
            "text": f"Error al buscar formación: {str(e)}",
            "type": "formation"
        }




@tool
def football_stats_analyst(question: str) -> str:
    """
    Herramienta avanzada de análisis de datos. 
    Su uso es efectivo cuando el usuario pregunte por estadísticas, goles, asistencias, 
    comparaciones entre jugadores o datos numéricos de la temporada 2024-2025.
    
    Args:
        question: La pregunta completa del usuario en lenguaje natural.
                  Ejemplo: "¿Quién es el máximo goleador del Arsenal?"
    
    Returns:
        La respuesta analizada basada en los datos reales.
    """
    try:
        logger.info(f"[stats_tool] Procesando pregunta: {question}")

        # Invocamos al agente SQL interno
        # Flujo: Pregunta -> Pensamiento -> Generar SQL -> Ejecutar -> Respuesta Final
        result = internal_sql_agent.invoke({"input": question})

        response = result.get('output', "No se pudo generar una respuesta.")
        logger.info(f"[stats_tool] Respuesta generada: {response}")

        return response

    except Exception as e:
        logger.error(f"[stats_tool] Error: {e}")
        return f"Error al consultar los datos: {str(e)}"
    

    