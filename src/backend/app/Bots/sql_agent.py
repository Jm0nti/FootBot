import logging
import kagglehub
import os
from langchain_core.messages import SystemMessage, AIMessage
from app.Bots.models import llm_smart
from app.Bots.tools import sql_executor
from app.Bots.types import AgentState
from sqlalchemy import create_engine
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from kagglehub import KaggleDatasetAdapter


logger = logging.getLogger(__name__)

# CARGA DE DATOS DESDE KAGGLE PARA EL SQL AGENT

DB_PATH = "football.db"
DB_URL = f"sqlite:///{DB_PATH}"

logger.info("--- INICIANDO CARGA DE DATOS ---")

# Si el archivo YA existe, NO volvemos a descargar ni regenerar
if os.path.exists(DB_PATH):
    logger.info("La base de datos ya existe. Usando archivo existente.")
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
    db = SQLDatabase(engine=engine)

else:
    logger.info("La base de datos no existe. Creándola por primera vez...")

    try:
        logger.info("Descargando dataset desde Kaggle...")

        df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "hubertsidorowicz/football-players-stats-2024-2025",
        "players_data-2024_2025.csv"
        )

        # Seleccionamos solo lo necesario para responder preguntas comunes de fútbol.

        cols_to_keep = [
            # Identidad
            'Player', 'Squad', 'Nation', 'Pos', 'Comp', 'Age',
            # Tiempo de Juego
            'MP', 'Starts', 'Min',
            # Ataque (Goles/Asistencias)
            'Gls', 'Ast', 'PK', 'PKatt', 
            # Métricas Avanzadas
            'xG', 'xAG', 
            # Disparos
            'Sh', 'SoT',
            # Defensa (Acciones defensivas)
            'Tkl', 'Int', 'Blocks', 'Clr',
            # Disciplina
            'CrdY', 'CrdR',
            # Porteros 
            'Saves', 'CS', 'GA' 
        ]

        # Filtrar solo si las columnas existen (intersección) para evitar errores
        existing_cols = [c for c in cols_to_keep if c in df.columns]
        df_optimized = df[existing_cols].copy()

        rename_map = {
            'Player': 'player_name',
            'Squad': 'team',
            'Nation': 'nationality',
            'Pos': 'position',
            'Comp': 'league',
            'Age': 'age',
            'MP': 'matches_played',
            'Starts': 'matches_started',
            'Min': 'minutes_played',
            'Gls': 'goals',
            'Ast': 'assists',
            'PK': 'penalties_scored',
            'PKatt': 'penalties_attempted',
            'xG': 'expected_goals',
            'xAG': 'expected_assisted_goals',
            'Sh': 'total_shots',
            'SoT': 'shots_on_target',
            'Tkl': 'tackles',
            'Int': 'interceptions',
            'Blocks': 'blocks',
            'Clr': 'clearances',
            'CrdY': 'yellow_cards',
            'CrdR': 'red_cards',
            'Saves': 'gk_saves',
            'CS': 'gk_clean_sheets',
            'GA': 'gk_goals_against'
        }

        df_optimized.rename(columns=rename_map, inplace=True)

        # LIMPIEZA DE NULOS
        numeric_cols = df_optimized.select_dtypes(include=['number']).columns
        df_optimized[numeric_cols] = df_optimized[numeric_cols].fillna(0)

        logger.info(f"Dataset podado. Registros: {len(df_optimized)}. Columnas finales: {list(df_optimized.columns)}")

        # Crear Motor SQL en archivo, no en memoria
        engine = create_engine(
            DB_URL,
            connect_args={"check_same_thread": False}
        )

        df_optimized.to_sql(name="players", con=engine, index=False, if_exists="replace")

        db = SQLDatabase(engine=engine)

        logger.info("Base de datos en memoria lista. Tabla 'players' creada.")

    except Exception as e:
        logger.exception("Error crítico cargando datos de Kaggle: %s", e)
        # Crear una DB vacía o manejar el error para que no rompa el script completo
        engine = create_engine("sqlite:///:memory:")
        db = SQLDatabase(engine=engine)



# DEFINICION DEL AGENTE SQL INTERNO

# Creamos el agente SQL una sola vez para reutilizarlo dentro de la tool.

schema_context = db.get_table_info()

sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm_smart)

system_message_sql = f"""
ROL:
Eres un Agente Experto en SQL (dialecto SQLite) y Analista Deportivo Senior.
Tu base de conocimientos son las estadísticas de fútbol 2024-2025.

ESQUEMA DE LA BASE DE DATOS (YA CONOCIDO):
---------------------------------------------------
{schema_context}
---------------------------------------------------

OBJETIVO:
Responder preguntas de negocio generando consultas SQL precisas basadas en el esquema de arriba.

REGLAS CRÍTICAS DE SQL (SQLite):
1. BÚSQUEDA FLEXIBLE: Usa SIEMPRE `WHERE LOWER(columna) LIKE '%valor%'` para nombres de jugadores o equipos.
   - Ejemplo: Si buscan "Haaland", usa `WHERE LOWER(player) LIKE '%haaland%'`.
2. NO ALUCINES COLUMNAS: Solo usa las columnas listadas en el ESQUEMA de arriba.
   - Fíjate bien si las columnas usan abreviaturas (ej: 'gls', 'ast', 'team').
3. LÍMITES: Agrega `LIMIT 5` para rankings.
4. ORDEN: Usa `ORDER BY columna DESC` para listas de "mejores" o "máximos".

CRITERIOS DE NEGOCIO:
- Si preguntan "quién es el mejor" (sin métrica), asume (Goles + Asistencias).
- Si no encuentras resultados, sugiere revisar el nombre.

FORMATO FINAL:
Responde en español natural resumiendo los datos encontrados.
"""

internal_sql_agent = create_sql_agent(
    llm=llm_smart,
    toolkit=sql_toolkit,
    verbose=True,
    handle_parsing_errors=True,
    prefix=system_message_sql
)

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


def sql_agent_node(state: AgentState) -> dict:
    """
    Nodo orquestador para estadísticas.
    Usa football_stats_analyst directamente para consultas SQL rápidas.
    """
    state.setdefault('trace', []).append('sql_agent')
    
    # 1. Obtenemos la última pregunta del usuario
    last_message = state['messages'][-1]
    user_query = last_message.content
    
    logger.info(f"[sql_node] Modo Directo activado para: {user_query}")
    

    try:
        tool_result = football_stats_analyst.invoke(user_query)
        
        final_text = str(tool_result)
        
    except Exception as e:
        logger.error(f"[sql_node] Error ejecutando tool directa: {e}")
        final_text = "Lo siento, tuve un problema técnico al consultar la base de datos de estadísticas."

    # 3. Construimos la respuesta final
    final_response = AIMessage(content=final_text)
    
    return {
        "messages": state['messages'] + [final_response],
        "needs_critic": True,
        "next_step": "critic",
        "trace": state.get('trace'),
        "origin": "sql_agent",
    }