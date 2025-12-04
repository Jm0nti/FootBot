from datetime import datetime
import pandas as pd
from langchain.agents import Tool


def _parse_time(s: str):
    """Parse time string into datetime.time object"""
    s = str(s).strip()
    formats = ["%H:%M", "%H.%M", "%H:%M:%S"]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).time()
        except Exception:
            pass
    if ":" in s:
        try:
            h, m = s.split(":")
            return datetime.strptime(f"{int(h):02d}:{int(m):02d}", "%H:%M").time()
        except Exception:
            pass
    return None


def _calcular_minutos_hasta(hora_bus):
    """Calcula minutos desde ahora hasta la hora del bus"""
    ahora = datetime.now()
    bus_datetime = datetime.combine(ahora.date(), hora_bus)
    return int((bus_datetime - ahora).total_seconds() / 60)


def make_tools_for_docs(df: pd.DataFrame):
    """
    Crea herramientas para consultar horarios de buses desde un DataFrame.
    
    Args:
        df: DataFrame con columnas 'Lugar de Partida', 'Hora de Partida', 'Lugar de Destino'
    
    Returns:
        Lista de Tools para usar con LangChain agents
    """
    
    # Preprocesar: parsear todas las horas una sola vez
    df = df.copy()
    df['hora_parsed'] = df['Hora de Partida'].apply(_parse_time)
    df = df.dropna(subset=['hora_parsed'])  # Eliminar filas con horas inválidas
    
    # Normalizar nombres de lugares para búsquedas más flexibles
    df['origen_lower'] = df['Lugar de Partida'].str.lower().str.strip()
    df['destino_lower'] = df['Lugar de Destino'].str.lower().str.strip()
    
    # Obtener lugares únicos para mensajes
    lugares_partida = sorted(df['Lugar de Partida'].unique())
    lugares_destino = sorted(df['Lugar de Destino'].unique())
    
    # Definir horarios fijos de primeros y últimos buses
    PRIMEROS_BUSES = {
        "rio": "5:35",
        "río": "5:35",
        "volador": "5:35",
        "minas": "5:50",
        "facultad de minas": "5:50"
    }
    
    ULTIMOS_BUSES = {
        "rio": "17:45",
        "río": "17:45",
        "volador": "17:50",
        "minas": "18:05",
        "facultad de minas": "18:05"
    }
    
    def siguiente_bus_tool(input_str: str) -> str:
        """
        Encuentra el próximo bus disponible.
        
        Input esperado: nombre del lugar de origen (ej: "volador", "minas")
        o formato clave=valor (ej: "origen=volador")
        """
        UMBRAL_MINUTOS = 5
        origen = None
        
        # Parsear input
        if input_str and input_str.strip():
            input_str = input_str.strip().lower()
            
            # Intentar parseo clave=valor
            if "=" in input_str:
                parts = [p.strip() for p in input_str.split(";")]
                for p in parts:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        if k.strip() in ("origen", "from", "lugar"):
                            origen = v.strip()
            else:
                # Verificar si el input coincide con algún lugar conocido
                lugares_conocidos = ["volador", "facultad de minas", "minas", "rio", "río"]
                if any(lugar in input_str for lugar in lugares_conocidos):
                    origen = input_str
        
        # Si no hay origen Y hay múltiples lugares de partida, preguntar
        if not origen and len(lugares_partida) > 1:
            lugares_str = ", ".join(lugares_partida)
            return f"¿Desde qué campus necesitas el bus? Puedo darte horarios desde: {lugares_str}."
        
        # Filtrar por hora actual
        ahora = datetime.now().time()
        df_futuros = df[df['hora_parsed'] > ahora].copy()
        
        # Filtrar por origen solo si se especificó
        if origen:
            df_futuros = df_futuros[df_futuros['origen_lower'].str.contains(origen, na=False)]
        
        if df_futuros.empty:
            origen_msg = f" desde {origen}" if origen else ""
            return (
                f"Aparentemente ya no quedan más buses intercampus{origen_msg} por hoy."
                "\n\nNota: Si te refieres a los buses que parten hacia el metro, "
                "puedes consultar los horarios completos aquí: "
                "https://medellin.unal.edu.co/images/sede_medio/2025/intercampus_2025_2/20250819RutasIntercampus2sAF.pdf"
            )
        
        # Ordenar por hora
        df_futuros = df_futuros.sort_values('hora_parsed')
        
        # Tomar el primero (próximo bus)
        primer_bus = df_futuros.iloc[0]
        hora_salida = primer_bus['hora_parsed']
        lugar_origen = primer_bus['Lugar de Partida']
        destino = primer_bus['Lugar de Destino']
        minutos = _calcular_minutos_hasta(hora_salida)
        
        # Si el bus está muy cerca, mencionar el siguiente también
        if minutos <= UMBRAL_MINUTOS and len(df_futuros) > 1:
            segundo_bus = df_futuros.iloc[1]
            hora_segundo = segundo_bus['hora_parsed']
            destino_segundo = segundo_bus['Lugar de Destino']
            minutos_segundo = _calcular_minutos_hasta(hora_segundo)
            
            if minutos <= 0:
                msg_principal = f"El bus de las {hora_salida.strftime('%H:%M')} hacia {destino} ya salió."
            else:
                msg_principal = (
                    f"El próximo bus desde {lugar_origen} sale en {minutos} minuto"
                    f"{'s' if minutos != 1 else ''} ({hora_salida.strftime('%H:%M')}) rumbo a {destino}."
                )
            
            msg_segundo = (
                f" Después tienes otro a las {hora_segundo.strftime('%H:%M')} hacia {destino_segundo}, "
                f"que pasa en {minutos_segundo} minutos."
            )
            return msg_principal + msg_segundo
        
        # Respuesta normal
        return (
            f"El próximo bus desde {lugar_origen} es a las {hora_salida.strftime('%H:%M')} hacia {destino}. "
            f"Llega en {minutos} minutos."
        )
    
    def primer_bus_tool(input_str: str) -> str:
        """
        Encuentra el primer bus del día desde un campus.
        
        Input esperado: nombre del lugar de origen (ej: "volador", "minas", "rio")
        """
        origen = None
        
        # Parsear input
        if input_str and input_str.strip():
            input_str = input_str.strip().lower()
            
            # Intentar parseo clave=valor
            if "=" in input_str:
                parts = [p.strip() for p in input_str.split(";")]
                for p in parts:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        if k.strip() in ("origen", "from", "lugar"):
                            origen = v.strip()
            else:
                # Asumir que es el nombre directo del lugar
                origen = input_str
        
        # Si no hay origen, preguntar
        if not origen:
            lugares_str = ", ".join(lugares_partida)
            return f"¿Desde qué campus quieres saber el primer bus? Opciones: {lugares_str}."
        
        # Buscar en el diccionario de primeros buses
        hora_primer_bus = None
        lugar_formal = None
        
        for key, hora in PRIMEROS_BUSES.items():
            if key in origen:
                hora_primer_bus = hora
                # Buscar el nombre formal del lugar en el DataFrame
                for lugar in lugares_partida:
                    if key in lugar.lower():
                        lugar_formal = lugar
                        break
                break
        
        if not hora_primer_bus:
            return (
                f"No encontré información del primer bus desde '{origen}'. "
                f"Los campus disponibles son: {', '.join(lugares_partida)}."
            )
        
        # Obtener destino del primer bus desde el DataFrame
        hora_parsed = _parse_time(hora_primer_bus)
        primer_bus_df = df[
            (df['origen_lower'].str.contains(origen, na=False)) & 
            (df['hora_parsed'] == hora_parsed)
        ]
        
        if not primer_bus_df.empty:
            destino = primer_bus_df.iloc[0]['Lugar de Destino']
            lugar_formal = primer_bus_df.iloc[0]['Lugar de Partida']
        else:
            destino = "el otro campus"
        
        respuesta = f"El primer bus desde {lugar_formal} sale a las {hora_primer_bus} hacia {destino}."
        
        # Agregar nota sobre buses al metro si es relevante
        respuesta += (
            "\n\nNota: Si te refieres a los buses con que parten desde el metro, "
            "puedes consultar los horarios completos aquí: "
            "https://medellin.unal.edu.co/images/sede_medio/2025/intercampus_2025_2/20250819RutasIntercampus2sAF.pdf"
        )
        
        return respuesta
    
    def ultimo_bus_tool(input_str: str) -> str:
        """
        Encuentra el último bus del día desde un campus.
        
        Input esperado: nombre del lugar de origen (ej: "volador", "minas", "rio")
        """
        origen = None
        
        # Parsear input
        if input_str and input_str.strip():
            input_str = input_str.strip().lower()
            
            # Intentar parseo clave=valor
            if "=" in input_str:
                parts = [p.strip() for p in input_str.split(";")]
                for p in parts:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        if k.strip() in ("origen", "from", "lugar"):
                            origen = v.strip()
            else:
                # Asumir que es el nombre directo del lugar
                origen = input_str
        
        # Si no hay origen, preguntar
        if not origen:
            lugares_str = ", ".join(lugares_partida)
            return f"¿Desde qué campus quieres saber el último bus? Opciones: {lugares_str}."
        
        # Buscar en el diccionario de últimos buses
        hora_ultimo_bus = None
        lugar_formal = None
        
        for key, hora in ULTIMOS_BUSES.items():
            if key in origen:
                hora_ultimo_bus = hora
                # Buscar el nombre formal del lugar en el DataFrame
                for lugar in lugares_partida:
                    if key in lugar.lower():
                        lugar_formal = lugar
                        break
                break
        
        if not hora_ultimo_bus:
            return (
                f"No encontré información del último bus desde '{origen}'. "
                f"Los campus disponibles son: {', '.join(lugares_partida)}."
            )
        
        # Obtener destino del último bus desde el DataFrame
        hora_parsed = _parse_time(hora_ultimo_bus)
        ultimo_bus_df = df[
            (df['origen_lower'].str.contains(origen, na=False)) & 
            (df['hora_parsed'] == hora_parsed)
        ]
        
        if not ultimo_bus_df.empty:
            destino = ultimo_bus_df.iloc[0]['Lugar de Destino']
            lugar_formal = ultimo_bus_df.iloc[0]['Lugar de Partida']
        else:
            destino = "el otro campus"
        
        # Calcular cuánto falta para el último bus
        ahora = datetime.now().time()
        hora_ultimo_parsed = _parse_time(hora_ultimo_bus)
        
        if ahora < hora_ultimo_parsed:
            minutos = _calcular_minutos_hasta(hora_ultimo_parsed)
            tiempo_msg = f" (en {minutos} minutos)" if minutos > 0 else ""
        else:
            tiempo_msg = " (este bus ya salió por hoy)"
        
        respuesta = f"El último bus desde {lugar_formal} sale a las {hora_ultimo_bus} hacia {destino}{tiempo_msg}."
        
        # Agregar nota sobre buses al metro si es relevante
        respuesta += (
            "\n\nNota: Si te refieres a los buses con dirección al metro, "
            "puedes consultar los horarios completos aquí: "
            "https://medellin.unal.edu.co/images/sede_medio/2025/intercampus_2025_2/20250819RutasIntercampus2sAF.pdf"
        )
        
        return respuesta
    
    def horarios_completos_tool(input_str) -> str:
        """
        Proporciona un enlace a los horarios completos de buses intercampus.
        """
        return (
            "Puedes consultar los horarios completos de los buses intercampus aquí: "
            "https://medellin.unal.edu.co/images/sede_medio/2025/intercampus_2025_2/20250819RutasIntercampus2sAF.pdf"
        )
    
    # Crear las herramientas
    tools = [
        Tool(
            name="SiguienteBus",
            func=siguiente_bus_tool,
            description=(
                "Encuentra el próximo bus intercampus disponible. "
                "Input: el campus de partida como string simple (ejemplos: 'volador', 'minas', 'rio'). "
                "Si el usuario no especifica el campus, pasa un string vacío '' y la herramienta preguntará. "
                "Usa esta tool cuando pregunten por el siguiente bus, próximo bus, o a qué hora sale el bus."
            )
        ),
        Tool(
            name="PrimerBus",
            func=primer_bus_tool,
            description=(
                "Encuentra el PRIMER bus del día desde un campus. "
                "Input: el campus de partida como string simple (ejemplos: 'volador', 'minas', 'rio'). "
                "Si el usuario no especifica el campus, pasa un string vacío '' y la herramienta preguntará. "
                "Usa esta tool cuando pregunten específicamente por el primer bus del día, el bus más temprano, "
                "o el bus de la madrugada."
            ),
            return_direct=True
        ),
        Tool(
            name="UltimoBus",
            func=ultimo_bus_tool,
            description=(
                "Encuentra el ÚLTIMO bus del día desde un campus. "
                "Input: el campus de partida como string simple (ejemplos: 'volador', 'minas', 'rio'). "
                "Si el usuario no especifica el campus, pasa un string vacío '' y la herramienta preguntará. "
                "Usa esta tool cuando pregunten específicamente por el último bus del día, el bus más tarde, "
                "el bus de la noche, o hasta qué hora hay buses."
            ),
            return_direct=True
        ),
        Tool(
            name="HorariosCompletos",
            func=horarios_completos_tool,
            description=(
                "Proporciona el enlace al PDF oficial con todos los horarios de buses intercampus. "
                "Usa esta herramienta cuando el usuario pida: 'horarios completos', 'todos los horarios', "
                "'dame el PDF', '¿tienes los horarios?', 'horarios de buses', o cuando quiera ver "
                "toda la información de horarios disponible."
            ),
            return_direct=True
        )
    ]
    
    return tools