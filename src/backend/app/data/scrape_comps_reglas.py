"""
Script para descargar información de competiciones de fútbol y reglamentos 
desde Wikipedia y almacenarla en archivos .txt para luego crear una vector DB en FAISS
"""

import wikipediaapi
import os
import time
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Lista organizada de competiciones y conceptos del fútbol
COMPETICIONES_Y_REGLAS = [
    
    # ========== COMPETICIONES INTERNACIONALES DE SELECCIONES (8) ==========
    "Copa Mundial de Fútbol",
    "Eurocopa",
    "Copa América",
    "Copa Africana de Naciones",
    "Copa de Oro de la Concacaf",
    "Copa Asiática",
    "Copa Confederaciones",
    "Liga de Naciones de la UEFA",
    
    # ========== COMPETICIONES INTERNACIONALES DE CLUBES (6) ==========
    "Liga de Campeones de la UEFA",
    "Liga Europa de la UEFA",
    "Liga Conferencia Europa de la UEFA",
    "Copa Libertadores",
    "Copa Sudamericana",
    "Mundial de Clubes de la FIFA",
    
    # ========== LIGAS NACIONALES EUROPA (10) ==========
    "Primera División de España",
    "Premier League",
    "Serie A",
    "Bundesliga",
    "Ligue 1",
    "Primeira Liga",
    "Eredivisie",
    "Liga Belga de Fútbol",
    "Süper Lig",
    "Premier League de Escocia",
    
    # ========== LIGAS NACIONALES AMÉRICA (8) ==========
    "Categoría Primera A",  # Colombia (Dimayor)
    "Campeonato Brasileiro de Futebol Serie A",
    "Liga MX",
    "Major League Soccer",
    "Primera División de Argentina",
    "Campeonato Uruguayo de Fútbol",
    "Campeonato Nacional de Fútbol de Chile",
    "Liga Profesional Venezolana de Fútbol",
    
    # ========== COPAS NACIONALES IMPORTANTES (6) ==========
    "Copa del Rey",
    "FA Cup",
    "Copa de Italia",
    "Copa de Alemania",
    "Copa de Francia",
    "Copa de Brasil",
    
    # ========== REGLAS Y REGLAMENTOS (15) ==========
    "Reglas de juego del fútbol",
    "Fuera de juego (fútbol)",
    "Sistema de videoarbitraje",  # VAR
    "Tarjeta amarilla y roja",
    "Penalti (fútbol)",
    "Tiro libre (fútbol)",
    "Saque de esquina",
    "Tiempo extra",
    "Tanda de penaltis",
    "Regla del gol de oro",
    "Regla del gol de plata",
    "Mano (fútbol)",
    "Falta (fútbol)",
    "Árbitro de fútbol",
    "Árbitro asistente de video",
    
    # ========== CONCEPTOS TÁCTICOS Y TÉCNICOS (12) ==========
    "Formación (fútbol)",
    "Sistema táctico",
    "Contraataque",
    "Pressing (fútbol)",
    "Posesión de balón",
    "Regate",
    "Pase (fútbol)",
    "Centro (fútbol)",
    "Cabezazo",
    "Disparo (fútbol)",
    "Parada (fútbol)",
    "Marcaje (fútbol)",
    
    # ========== POSICIONES Y ROLES (11) ==========
    "Portero (fútbol)",
    "Defensa (fútbol)",
    "Lateral (fútbol)",
    "Defensa central",
    "Líbero",
    "Centrocampista",
    "Mediapunta",
    "Pivote (fútbol)",
    "Extremo (fútbol)",
    "Delantero centro",
    "Segundo delantero",
    
    # ========== PREMIOS Y RECONOCIMIENTOS (6) ==========
    "Balón de Oro",
    "The Best FIFA Football Awards",
    "Bota de Oro",
    "Guante de Oro",
    "Jugador Joven del Año de la FIFA",
    "Premio Puskás"
]

# Directorio donde se guardarán las informaciones
OUTPUT_DIR = "competiciones_y_reglas"

# Configuración de Wikipedia API
USER_AGENT = 'Football-Competitions-Rules-Scraper/1.0'

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


def search_topic(wiki, topic_name):
    """
    Busca un tema en Wikipedia
    
    Args:
        wiki: Objeto Wikipedia API
        topic_name: Nombre del tema a buscar
    
    Returns:
        Nombre correcto de la página de Wikipedia o None
    """
    try:
        # Intentar con el nombre exacto
        page = wiki.page(topic_name)
        if page.exists():
            return topic_name
        
        # Si no funciona, intentar variaciones
        print(f"   → Buscando variaciones de '{topic_name}'...")
        
        variations = [
            topic_name,
            topic_name.replace("de la UEFA", ""),
            topic_name.replace("de la FIFA", ""),
            topic_name + " de fútbol",
            topic_name.replace("(fútbol)", "")
        ]
        
        for variation in variations:
            page = wiki.page(variation.strip())
            if page.exists():
                print(f"   ✓ Encontrado como: '{variation}'")
                return variation.strip()
        
        return None
        
    except Exception as e:
        print(f"   ✗ Error buscando '{topic_name}': {str(e)}")
        return None


def download_topic_info(wiki, topic_name, output_dir):
    """
    Descarga la información de un tema y la guarda en un archivo .txt
    
    Args:
        wiki: Objeto Wikipedia API
        topic_name: Nombre del tema
        output_dir: Directorio donde guardar el archivo
    
    Returns:
        True si se descargó exitosamente, False en caso contrario
    """
    try:
        # Buscar el tema
        correct_name = search_topic(wiki, topic_name)
        
        if not correct_name:
            print(f"❌ No se encontró: {topic_name}")
            return False
        
        # Obtener la página
        page = wiki.page(correct_name)
        
        if not page.exists():
            print(f"❌ Página no existe: {topic_name}")
            return False
        
        # Extraer información
        title = page.title
        summary = page.summary
        full_text = page.text
        
        # Crear nombre de archivo seguro
        safe_filename = "".join(c for c in topic_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_filename = safe_filename.replace(' ', '_')
        safe_filename = safe_filename.replace('(', '').replace(')', '')
        filepath = os.path.join(output_dir, f"{safe_filename}.txt")
        
        # Guardar en archivo
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"TEMA: {title}\n")
            f.write("=" * 80 + "\n\n")
            f.write("RESUMEN:\n")
            f.write("-" * 80 + "\n")
            f.write(summary + "\n\n")
            f.write("INFORMACIÓN COMPLETA:\n")
            f.write("-" * 80 + "\n")
            f.write(full_text)
        
        print(f"✅ Descargado: {topic_name} → {safe_filename}.txt")
        return True
        
    except Exception as e:
        print(f"❌ Error descargando {topic_name}: {str(e)}")
        return False


def download_all_topics(topics, output_dir=OUTPUT_DIR, delay=1.0):
    """
    Descarga toda la información de la lista de temas en español
    
    Args:
        topics: Lista de nombres de temas
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
        'total': len(topics),
        'temas_descargados': [],
        'temas_fallidos': []
    }
    
    print("=" * 80)
    print(f"DESCARGANDO INFORMACIÓN DE {len(topics)} COMPETICIONES Y REGLAS")
    print(f"Idioma: ESPAÑOL")
    print(f"Directorio: {output_dir}")
    print("=" * 80 + "\n")
    
    # Descargar cada tema
    for i, topic in enumerate(topics, 1):
        print(f"\n[{i}/{len(topics)}] Procesando: {topic}")
        
        success = download_topic_info(wiki, topic, output_dir)
        
        if success:
            stats['exitosos'] += 1
            stats['temas_descargados'].append(topic)
        else:
            stats['fallidos'] += 1
            stats['temas_fallidos'].append(topic)
        
        # Esperar para evitar rate limiting
        if i < len(topics):
            time.sleep(delay)
    
    # Imprimir resumen final
    print("\n" + "=" * 80)
    print("RESUMEN DE DESCARGA")
    print("=" * 80)
    print(f"✅ Exitosos: {stats['exitosos']}/{stats['total']}")
    print(f"❌ Fallidos: {stats['fallidos']}/{stats['total']}")
    print(f"📊 Tasa de éxito: {(stats['exitosos']/stats['total']*100):.1f}%")
    
    if stats['temas_fallidos']:
        print(f"\n⚠️ Temas no encontrados:")
        for topic in stats['temas_fallidos']:
            print(f"   - {topic}")
    
    print(f"\n📁 Archivos guardados en: {output_dir}/")
    print("=" * 80)
    
    return stats


def create_metadata_file(stats, output_dir=OUTPUT_DIR):
    """
    Crea un archivo de metadatos con la información de descarga
    """
    metadata_path = os.path.join(output_dir, "_metadata.txt")
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write("METADATOS DE DESCARGA - COMPETICIONES Y REGLAS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total temas: {stats['total']}\n")
        f.write(f"Descargados exitosamente: {stats['exitosos']}\n")
        f.write(f"Fallidos: {stats['fallidos']}\n\n")
        
        f.write("TEMAS DESCARGADOS:\n")
        f.write("-" * 80 + "\n")
        for topic in stats['temas_descargados']:
            f.write(f"✓ {topic}\n")
        
        if stats['temas_fallidos']:
            f.write("\nTEMAS NO ENCONTRADOS:\n")
            f.write("-" * 80 + "\n")
            for topic in stats['temas_fallidos']:
                f.write(f"✗ {topic}\n")
    
    print(f"📝 Metadatos guardados en: {metadata_path}")


# ============================================================================
# FUNCIÓN PARA MOSTRAR LA LISTA COMPLETA
# ============================================================================

def print_topics_list():
    """
    Imprime la lista organizada de temas por categoría
    """
    print("\n" + "=" * 80)
    print("LISTA COMPLETA DE COMPETICIONES Y REGLAS A DESCARGAR")
    print("=" * 80 + "\n")
    
    categories = {
        "🌍 COMPETICIONES INTERNACIONALES - SELECCIONES": COMPETICIONES_Y_REGLAS[0:8],
        "🏆 COMPETICIONES INTERNACIONALES - CLUBES": COMPETICIONES_Y_REGLAS[8:14],
        "🇪🇺 LIGAS EUROPEAS": COMPETICIONES_Y_REGLAS[14:24],
        "🌎 LIGAS AMERICANAS": COMPETICIONES_Y_REGLAS[24:32],
        "🏅 COPAS NACIONALES": COMPETICIONES_Y_REGLAS[32:38],
        "📋 REGLAS Y REGLAMENTOS": COMPETICIONES_Y_REGLAS[38:53],
        "⚽ CONCEPTOS TÁCTICOS Y TÉCNICOS": COMPETICIONES_Y_REGLAS[53:65],
        "👥 POSICIONES Y ROLES": COMPETICIONES_Y_REGLAS[65:76],
        "🏅 PREMIOS Y RECONOCIMIENTOS": COMPETICIONES_Y_REGLAS[76:82]
    }
    
    total = 0
    for category, topics in categories.items():
        print(f"{category} - {len(topics)} temas")
        print("-" * 80)
        for topic in topics:
            print(f"  • {topic}")
            total += 1
        print()
    
    print("=" * 80)
    print(f"TOTAL: {total} temas")
    print("=" * 80 + "\n")


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    """
    Ejecutar el script para descargar toda la información de competiciones 
    y reglas en español
    """
    
    print("\n🏆 SCRAPER DE COMPETICIONES Y REGLAS - WIKIPEDIA")
    print("=" * 80 + "\n")
    
    # Mostrar lista de temas
    print("¿Deseas ver la lista completa de temas antes de descargar?")
    show_list = input("(s/n, default=n): ").strip().lower()
    
    if show_list == 's':
        print_topics_list()
    
    # Confirmar inicio
    input(f"Se descargará información de {len(COMPETICIONES_Y_REGLAS)} temas en ESPAÑOL. Presiona ENTER para comenzar...")
    
    # Descargar información
    stats = download_all_topics(
        topics=COMPETICIONES_Y_REGLAS,
        output_dir=OUTPUT_DIR,
        delay=1.0  # 1 segundo entre requests
    )
    
    # Crear archivo de metadatos
    create_metadata_file(stats, OUTPUT_DIR)
    
    print("\n✨ Proceso completado.")