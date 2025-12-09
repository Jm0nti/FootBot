import logging
import kagglehub
import os
import pandas as pd
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

# ============================================================================
# CARGA Y PROCESAMIENTO DE DATOS (ACTUAL + HISTÓRICO)
# ============================================================================

DB_PATH = "football.db"
DB_URL = f"sqlite:///{DB_PATH}"

logger.info("--- VERIFICANDO BASE DE DATOS ---")


if os.path.exists(DB_PATH):
    logger.info("La base de datos ya existe. Conectando...")
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
    db = SQLDatabase(engine=engine)

else:
    logger.info("La base de datos no existe. Iniciando descarga y procesamiento...")
    
    # Creamos el motor SQL
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False})

    # DATASET ACTUAL (2024-2025) 
    try:
        logger.info("Descargando dataset ACTUAL (24/25)...")
        df_current = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "hubertsidorowicz/football-players-stats-2024-2025",
            "players_data-2024_2025.csv"
        )
        
        # Selección de columnas actuales
        cols_current = [
            'Player', 'Squad', 'Nation', 'Pos', 'Comp', 'Age', # Info
            'MP', 'Starts', 'Min', # Tiempo
            'Gls', 'Ast', 'PK', 'PKatt', # Goles
            'xG', 'xAG', # Avanzadas
            'Sh', 'SoT', 'Tkl', 'Int', 'Blocks', 'Clr', # Juego
            'CrdY', 'CrdR', 'Saves', 'CS', 'GA' # Disciplina y Portero
        ]
        
        # Filtrado y Renombrado
        existing_cols = [c for c in cols_current if c in df_current.columns]
        df_curr_opt = df_current[existing_cols].copy()
        
        rename_map_curr = {
            'Player': 'player_name', 'Squad': 'team', 'Nation': 'nationality', 
            'Pos': 'position', 'Comp': 'league', 'Age': 'age',
            'MP': 'matches_played', 'Starts': 'matches_started', 'Min': 'minutes_played',
            'Gls': 'goals', 'Ast': 'assists', 'PK': 'penalties_scored',
            'xG': 'expected_goals', 'Sh': 'total_shots', 'SoT': 'shots_on_target',
            'Tkl': 'tackles', 'Int': 'interceptions', 'CrdY': 'yellow_cards', 
            'CrdR': 'red_cards', 'Saves': 'gk_saves', 'CS': 'gk_clean_sheets'
        }
        df_curr_opt.rename(columns=rename_map_curr, inplace=True)
        
        # Limpieza Nulos
        num_cols = df_curr_opt.select_dtypes(include=['number']).columns
        df_curr_opt[num_cols] = df_curr_opt[num_cols].fillna(0)
        
        logger.info(f"📊 Estructura DataFrame 'players' (Actual):")
        logger.info(f"   - Filas: {len(df_curr_opt)}")
        logger.info(f"   - Columnas ({len(df_curr_opt.columns)}): {list(df_curr_opt.columns)}")
        
        # Guardar tabla 'players' (ACTUAL)
        df_curr_opt.to_sql(name="players", con=engine, index=False, if_exists="replace")
        logger.info(f"Tabla 'players' (2024-2025) creada. {len(df_curr_opt)} registros.")

    except Exception as e:
        logger.error(f"Error cargando dataset actual: {e}")

    # --- DATASET HISTÓRICO (1992-2025) ---
    try:
        logger.info("Descargando dataset HISTÓRICO (1992-2025)...")
        
        # CAMBIO REALIZADO: Usando load_dataset directamente con el nombre del archivo
        df_hist = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "patryk060801/football-players-1992-2025-top-5-leagues",
            "All_Players_1992-2025.csv"
        )
        
        # Columnas clave para historia (incluyendo premios y temporadas)
        cols_hist = [
            'Player', 'Squad', 'Nation', 'Pos', 'League', 'Season', 'Age', # Identidad + Temporada
            'MP', 'Starts', 'Min', # Tiempo
            'Gls', 'Ast', 'PK', # Goles básicos
            'CrdY', 'CrdR', # Tarjetas
            'Ballon d\'or', 'UCL_Won', 'League Won', 'European Golden Shoe' # PREMIOS
        ]
        
        # Intersección segura
        existing_cols_hist = [c for c in cols_hist if c in df_hist.columns]
        df_hist_opt = df_hist[existing_cols_hist].copy()
        
        rename_map_hist = {
            'Player': 'player_name', 'Squad': 'team', 'Nation': 'nationality',
            'Pos': 'position', 'League': 'league', 'Season': 'season', 'Age': 'age',
            'MP': 'matches_played', 'Starts': 'matches_started', 'Min': 'minutes_played',
            'Gls': 'goals', 'Ast': 'assists', 'PK': 'penalties_scored',
            'CrdY': 'yellow_cards', 'CrdR': 'red_cards',
            'Ballon d\'or': 'ballon_dor_wins', 'UCL_Won': 'ucl_titles',
            'League Won': 'league_titles', 'European Golden Shoe': 'golden_shoe_wins'
        }
        df_hist_opt.rename(columns=rename_map_hist, inplace=True)
        
        # Limpieza básica
        num_cols_h = df_hist_opt.select_dtypes(include=['number']).columns
        df_hist_opt[num_cols_h] = df_hist_opt[num_cols_h].fillna(0)
        
        logger.info(f"📊 Estructura DataFrame 'historical_players':")
        logger.info(f"   - Filas: {len(df_hist_opt)}")
        logger.info(f"   - Columnas ({len(df_hist_opt.columns)}): {list(df_hist_opt.columns)}")
        
        # Guardar tabla 'historical_players'
        df_hist_opt.to_sql(name="historical_players", con=engine, index=False, if_exists="replace")
        logger.info(f"Tabla 'historical_players' creada. {len(df_hist_opt)} registros.")

    except Exception as e:
        logger.error(f"Error cargando dataset histórico: {e}")

    # Conectar DB 
    db = SQLDatabase(engine=engine)


# ============================================================================
# DEFINICIÓN DEL AGENTE SQL
# ============================================================================

schema_context = db.get_table_info()

sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm_smart)

# Prompt rediseñado para manejar DOS tablas y lógica temporal
system_message_sql = f"""
ROL:
Eres un Agente Experto en SQL (dialecto SQLite) y Analista Deportivo Senior.
Tienes acceso a DOS tablas de datos de fútbol. Debes elegir la correcta según la pregunta.

ESQUEMA DE BASE DE DATOS:
---------------------------------------------------
{schema_context}
---------------------------------------------------

TABLAS DISPONIBLES Y SU USO:
1. `players`: ÚSALA SOLO PARA LA TEMPORADA ACTUAL (2024-2025).
   - Preguntas sobre "actualidad", "esta temporada", "ahora", "hoy".
   - Preguntas sobre porteros.
   
2. `historical_players`: ÚSALA PARA HISTORIA (1992-2025).
   - Preguntas sobre "toda la carrera", "historia", "temporadas pasadas" (ej: 2012, 2015), o comparaciones históricas.
   - Contiene premios como 'ballon_dor_wins', 'ucl_titles'.
   - IMPORTANTE: Un jugador aparece MÚLTIPLES VECES (una fila por temporada). 
     - Si piden "total de goles en su carrera", usa `SUM(goals) GROUP BY player_name`.
     - Si piden "goles en 2012", filtra por `WHERE season LIKE '%2011-2012%'` (o similar).

REGLAS DE ORO SQL (SQLite):
1. BÚSQUEDA DE NOMBRES: SIEMPRE usa `WHERE LOWER(player_name) LIKE '%messi%'`.
2. FORMATO DE TEMPORADA: En `historical_players`, la columna `season` tiene formato "YYYY-YYYY" (ej: "2011-2012"). Usa `LIKE` para filtrar años.
3. PREMIOS: Si preguntan por Balones de Oro, usa `SUM(ballon_dor_wins)`.

CRITERIOS DE RESPUESTA:
- Si preguntan "¿Quién tiene más goles?", pregunta implícitamente "¿En la historia o ahora?". 
- Si no especifican, ASUME HISTORIA (`historical_players`) y agrúpalo por jugador.
- NO ALUCINES COLUMNAS: Solo usa las columnas listadas en el ESQUEMA de arriba.
- LÍMITES: Agrega `LIMIT 5` para rankings.
- ORDEN: Usa `ORDER BY columna DESC` para listas de "mejores" o "máximos".

CRITERIOS DE NEGOCIO:
- Si preguntan "quién es el mejor" (sin métrica), asume (Goles + Asistencias).
- Si no encuentras resultados, sugiere revisar el nombre.

FORMATO FINAL:
Responde en español natural, mencionando explícitamente si los datos son de esta temporada o históricos, si son datos historicos menciona que son extraidos del top
5 ligas europeas.
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
    Herramienta experta en estadísticas de fútbol (Actuales e Históricas).
    Úsala para consultar goles, asistencias, premios y datos desde 1992 hasta hoy.
    """
    try:
        logger.info(f"[stats_tool] Procesando: {question}")
        result = internal_sql_agent.invoke({"input": question})
        response = result.get('output', "No se pudo generar respuesta.")
        return response
    except Exception as e:
        logger.error(f"[stats_tool] Error: {e}")
        return f"Error técnico: {str(e)}"


# ============================================================================
# NODO ORQUESTADOR 
# ============================================================================

def sql_agent_node(state: AgentState) -> dict:
    """
    Nodo orquestador que conecta directo con el analista de datos.
    """
    state.setdefault('trace', []).append('sql_agent')
    last_message = state['messages'][-1]
    user_query = last_message.content
    
    logger.info(f"[sql_node] Consulta directa: {user_query}")
    
    try:
        # Ejecución directa
        tool_result = football_stats_analyst.invoke(user_query)
        final_text = str(tool_result)
        
    except Exception as e:
        logger.error(f"[sql_node] Error: {e}")
        final_text = "Lo siento, tuve un problema consultando la base de datos."

    final_response = AIMessage(content=final_text)
    
    return {
        "messages": state['messages'] + [final_response],
        "needs_critic": True,
        "next_step": "critic",
        "trace": state.get('trace'),
        "origin": "sql_agent",
    }
    
