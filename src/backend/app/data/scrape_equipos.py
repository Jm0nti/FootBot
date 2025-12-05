"""
Script para descargar información de equipos de fútbol desde Wikipedia
y almacenarla en archivos .txt para luego crear una vector DB en FAISS
"""

import wikipediaapi
import os
import time
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Lista de los 60 equipos más importantes del mundo con sus nombres en Wikipedia
EQUIPOS = [
    # ========== ESPAÑA - LA LIGA (10) ==========
    "Real Madrid Club de Fútbol",
    "Fútbol Club Barcelona",
    "Club Atlético de Madrid",
    "Sevilla Fútbol Club",
    "Valencia Club de Fútbol",
    "Athletic Club",
    "Real Betis Balompié",
    "Real Sociedad de Fútbol",
    "Villarreal Club de Fútbol",
    "Real Club Celta de Vigo",
    
    # ========== INGLATERRA - PREMIER LEAGUE (10) ==========
    "Manchester City Football Club",
    "Liverpool Football Club",
    "Arsenal Football Club",
    "Chelsea Football Club",
    "Manchester United Football Club",
    "Tottenham Hotspur Football Club",
    "Newcastle United Football Club",
    "Aston Villa Football Club",
    "Brighton & Hove Albion Football Club",
    "West Ham United Football Club",
    
    # ========== ITALIA - SERIE A (8) ==========
    "Football Club Internazionale Milano",
    "Associazione Calcio Milan",
    "Juventus Football Club",
    "Società Sportiva Calcio Napoli",
    "Associazione Sportiva Roma",
    "Società Sportiva Lazio",
    "Atalanta Bergamasca Calcio",
    "ACF Fiorentina",
    
    # ========== ALEMANIA - BUNDESLIGA (8) ==========
    "Bayern de Múnich",
    "Borussia Dortmund",
    "RB Leipzig",
    "Bayer 04 Leverkusen",
    "Borussia Mönchengladbach",
    "Eintracht Frankfurt",
    "VfB Stuttgart",
    "Wolfsburgo",
    
    # ========== FRANCIA - LIGUE 1 (6) ==========
    "Paris Saint-Germain Football Club",
    "Olympique de Marsella",
    "Olympique de Lyon",
    "AS Mónaco",
    "Lille OSC",
    "OGC Niza",
    
    # ========== PORTUGAL (4) ==========
    "Sport Lisboa e Benfica",
    "Futebol Clube do Porto",
    "Sporting de Lisboa",
    "Sporting Clube de Braga",
    
    # ========== PAÍSES BAJOS (3) ==========
    "Ajax de Ámsterdam",
    "PSV Eindhoven",
    "Feyenoord Rotterdam",
    
    # ========== SUDAMÉRICA (6) ==========
    "Club Atlético Boca Juniors",
    "Club Atlético River Plate",
    "Clube de Regatas do Flamengo",
    "São Paulo Futebol Clube",
    "Club Atlético Peñarol",
    "Club Nacional de Football",
    
    # ========== OTROS EUROPA (5) ==========
    "Celtic Football Club",
    "Rangers Football Club",
    "Galatasaray Spor Kulübü",
    "Fenerbahçe Spor Kulübü",
    "Shakhtar Donetsk"
]

# Directorio donde se guardarán las informaciones
OUTPUT_DIR = "informacion_equipos"

# Configuración de Wikipedia API
USER_AGENT = 'Football-Teams-Info-Scraper/1.0'

# ============================================================================
# FUNCIONES PRINCIPALES
# ============================================================================

def setup_wikipedia_api():
    """
    Configura la API de Wikipedia en español
    
    Returns:
        Objeto Wikipedia API configurado en español
    """
    return wikipediaapi.Wikipedia(
        user_agent=USER_AGENT,
        language='es',
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )


def search_team(wiki, team_name):
    """
    Busca un equipo en Wikipedia
    
    Args:
        wiki: Objeto Wikipedia API
        team_name: Nombre del equipo a buscar
    
    Returns:
        Nombre correcto de la página de Wikipedia o None
    """
    try:
        # Intentar con el nombre exacto
        page = wiki.page(team_name)
        if page.exists():
            # Verificar que sea un club de fútbol
            summary_lower = page.summary[:500].lower()
            keywords = ['fútbol', 'football', 'club', 'equipo', 'fundado', 
                       'estadio', 'liga', 'soccer', 'calcio']
            
            if any(keyword in summary_lower for keyword in keywords):
                return team_name
        
        # Si no funciona, intentar variaciones
        print(f"   → Buscando variaciones de '{team_name}'...")
        
        variations = [
            team_name,
            team_name.replace('Football Club', 'FC'),
            team_name.replace('Club de Fútbol', 'CF'),
            team_name.replace('Fútbol Club', 'FC'),
            team_name + " (fútbol)"
        ]
        
        for variation in variations:
            page = wiki.page(variation)
            if page.exists():
                summary_lower = page.summary[:500].lower()
                keywords = ['fútbol', 'football', 'club', 'equipo', 'estadio']
                
                if any(keyword in summary_lower for keyword in keywords):
                    print(f"   ✓ Encontrado como: '{variation}'")
                    return variation
        
        return None
        
    except Exception as e:
        print(f"   ✗ Error buscando '{team_name}': {str(e)}")
        return None


def download_team_info(wiki, team_name, output_dir):
    """
    Descarga la información de un equipo y la guarda en un archivo .txt
    
    Args:
        wiki: Objeto Wikipedia API
        team_name: Nombre del equipo
        output_dir: Directorio donde guardar el archivo
    
    Returns:
        True si se descargó exitosamente, False en caso contrario
    """
    try:
        # Buscar el equipo
        correct_name = search_team(wiki, team_name)
        
        if not correct_name:
            print(f"❌ No se encontró: {team_name}")
            return False
        
        # Obtener la página
        page = wiki.page(correct_name)
        
        if not page.exists():
            print(f"❌ Página no existe: {team_name}")
            return False
        
        # Extraer información
        title = page.title
        summary = page.summary
        full_text = page.text
        
        # Crear nombre de archivo seguro
        safe_filename = "".join(c for c in team_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_filename = safe_filename.replace(' ', '_')
        filepath = os.path.join(output_dir, f"{safe_filename}.txt")
        
        # Guardar en archivo
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"EQUIPO: {title}\n")
            f.write("=" * 80 + "\n\n")
            f.write("RESUMEN:\n")
            f.write("-" * 80 + "\n")
            f.write(summary + "\n\n")
            f.write("INFORMACIÓN COMPLETA:\n")
            f.write("-" * 80 + "\n")
            f.write(full_text)
        
        print(f"✅ Descargado: {team_name} → {safe_filename}.txt")
        return True
        
    except Exception as e:
        print(f"❌ Error descargando {team_name}: {str(e)}")
        return False


def download_all_teams(teams, output_dir=OUTPUT_DIR, delay=1.0):
    """
    Descarga toda la información de la lista de equipos en español
    
    Args:
        teams: Lista de nombres de equipos
        output_dir: Directorio donde guardar los archivos
        delay: Segundos de espera entre requests (para evitar rate limiting)
    
    Returns:
        Diccionario con estadísticas de descarga
    """
    # Crear directorio si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Configurar Wikipedia API en español
    wiki = setup_wikipedia_api()
    
    # Estadísticas
    stats = {
        'exitosos': 0,
        'fallidos': 0,
        'total': len(teams),
        'equipos_descargados': [],
        'equipos_fallidos': []
    }
    
    print("=" * 80)
    print(f"DESCARGANDO INFORMACIÓN DE {len(teams)} EQUIPOS DE FÚTBOL")
    print(f"Idioma: ESPAÑOL")
    print(f"Directorio: {output_dir}")
    print("=" * 80 + "\n")
    
    # Descargar cada equipo
    for i, team in enumerate(teams, 1):
        print(f"\n[{i}/{len(teams)}] Procesando: {team}")
        
        success = download_team_info(wiki, team, output_dir)
        
        if success:
            stats['exitosos'] += 1
            stats['equipos_descargados'].append(team)
        else:
            stats['fallidos'] += 1
            stats['equipos_fallidos'].append(team)
        
        # Esperar para evitar rate limiting
        if i < len(teams):
            time.sleep(delay)
    
    # Imprimir resumen final
    print("\n" + "=" * 80)
    print("RESUMEN DE DESCARGA")
    print("=" * 80)
    print(f"✅ Exitosos: {stats['exitosos']}/{stats['total']}")
    print(f"❌ Fallidos: {stats['fallidos']}/{stats['total']}")
    print(f"📊 Tasa de éxito: {(stats['exitosos']/stats['total']*100):.1f}%")
    
    if stats['equipos_fallidos']:
        print(f"\n⚠️ Equipos no encontrados:")
        for team in stats['equipos_fallidos']:
            print(f"   - {team}")
    
    print(f"\n📁 Archivos guardados en: {output_dir}/")
    print("=" * 80)
    
    return stats


def create_metadata_file(stats, output_dir=OUTPUT_DIR):
    """
    Crea un archivo de metadatos con la información de descarga
    """
    metadata_path = os.path.join(output_dir, "_metadata.txt")
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write("METADATOS DE DESCARGA - INFORMACIÓN DE EQUIPOS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total equipos: {stats['total']}\n")
        f.write(f"Descargados exitosamente: {stats['exitosos']}\n")
        f.write(f"Fallidos: {stats['fallidos']}\n\n")
        
        f.write("EQUIPOS DESCARGADOS:\n")
        f.write("-" * 80 + "\n")
        for team in stats['equipos_descargados']:
            f.write(f"✓ {team}\n")
        
        if stats['equipos_fallidos']:
            f.write("\nEQUIPOS NO ENCONTRADOS:\n")
            f.write("-" * 80 + "\n")
            for team in stats['equipos_fallidos']:
                f.write(f"✗ {team}\n")
    
    print(f"📝 Metadatos guardados en: {metadata_path}")


# ============================================================================
# FUNCIÓN PARA MOSTRAR LA LISTA COMPLETA
# ============================================================================

def print_teams_list():
    """
    Imprime la lista organizada de equipos por liga/región
    """
    print("\n" + "=" * 80)
    print("LISTA COMPLETA DE EQUIPOS A DESCARGAR")
    print("=" * 80 + "\n")
    
    leagues = {
        "🇪🇸 LA LIGA (España)": EQUIPOS[0:10],
        "🏴󠁧󠁢󠁥󠁮󠁧󠁿 PREMIER LEAGUE (Inglaterra)": EQUIPOS[10:20],
        "🇮🇹 SERIE A (Italia)": EQUIPOS[20:28],
        "🇩🇪 BUNDESLIGA (Alemania)": EQUIPOS[28:36],
        "🇫🇷 LIGUE 1 (Francia)": EQUIPOS[36:42],
        "🇵🇹 LIGA PORTUGUESA": EQUIPOS[42:46],
        "🇳🇱 EREDIVISIE (Países Bajos)": EQUIPOS[46:49],
        "🌎 SUDAMÉRICA": EQUIPOS[49:55],
        "🌍 OTROS EUROPA": EQUIPOS[55:60]
    }
    
    total = 0
    for league, teams in leagues.items():
        print(f"{league} - {len(teams)} equipos")
        print("-" * 80)
        for team in teams:
            print(f"  • {team}")
            total += 1
        print()
    
    print("=" * 80)
    print(f"TOTAL: {total} equipos")
    print("=" * 80 + "\n")


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    """
    Ejecutar el script para descargar toda la información de equipos en español
    """
    
    print("\n⚽ SCRAPER DE INFORMACIÓN DE EQUIPOS - WIKIPEDIA")
    print("=" * 80 + "\n")
    
    # Mostrar lista de equipos
    print("¿Deseas ver la lista completa de equipos antes de descargar?")
    show_list = input("(s/n, default=n): ").strip().lower()
    
    if show_list == 's':
        print_teams_list()
    
    # Confirmar inicio
    input(f"Se descargará información de {len(EQUIPOS)} equipos en ESPAÑOL. Presiona ENTER para comenzar...")
    
    # Descargar información
    stats = download_all_teams(
        teams=EQUIPOS,
        output_dir=OUTPUT_DIR,
        delay=1.0  # 1 segundo entre requests
    )
    
    # Crear archivo de metadatos
    create_metadata_file(stats, OUTPUT_DIR)
    
    print("\n✨ Proceso completado. La información de equipos está lista para:")
    print("   1. Crear embeddings")
    print("   2. Almacenar en FAISS junto con biografías de jugadores")
    print("   3. Usar en tu agente multiagente")
    print("\n¡Listo para el siguiente paso! 🚀\n")