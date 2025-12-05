"""
Script para descargar biografías de jugadores de fútbol desde Wikipedia
y almacenarlas en archivos .txt para luego crear una vector DB en FAISS
"""

import wikipediaapi
import os
import time
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

JUGADORES = [
    # Estrellas Actuales - Tier S
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Kylian Mbappé",
    "Erling Haaland",
    "Neymar",
    "Vinicius Junior",
    "Jude Bellingham",
    "Kevin De Bruyne",
    "Mohamed Salah",
    "Harry Kane",
    
    # Estrellas Actuales - Tier A
    "Robert Lewandowski",
    "Luka Modrić",
    "Karim Benzema",
    "Pedri",
    "Gavi",
    "Rodri",
    "Bukayo Saka",
    "Phil Foden",
    "Jamal Musiala",
    "Florian Wirtz",
    
    # Top Defensores y Porteros Actuales
    "Thibaut Courtois",
    "Alisson Becker",
    "Virgil van Dijk",
    "Rúben Dias",
    "Antonio Rüdiger",
    "Joško Gvardiol",
    "William Saliba",
    "Alphonso Davies",
    
    # Estrellas Consolidadas
    "Antoine Griezmann",
    "Luis Suárez",
    "Sergio Ramos",
    "Toni Kroos",
    "Casemiro",
    "Bruno Fernandes",
    "Son Heung-min",
    "Joshua Kimmich",
    "Bernardo Silva",
    "Federico Valverde",
    
    # Leyendas Históricas (Últimos 20 años)
    "Zinedine Zidane",
    "Ronaldinho",
    "Andrés Iniesta",
    "Xavi Hernández",
    "Sergio Busquets",
    "Gerard Piqué",
    "Iker Casillas",
    "Gianluigi Buffon",
    "Thierry Henry",
    "Wayne Rooney",
    "Zlatan Ibrahimović",
    "David Beckham",
    
    # ========== EXTENSIÓN: 50 JUGADORES ADICIONALES ==========
    
    # Jóvenes Promesas y Estrellas Emergentes
    "Lamine Yamal",
    "Endrick",
    "Alejandro Garnacho",
    "Arda Güler",
    "Warren Zaïre-Emery",
    "Xavi Simons",
    "Joao Neves",
    "Sávio",
    "Kobbie Mainoo",
    "Pau Cubarsí",
    
    # Delanteros de Elite Actuales
    "Victor Osimhen",
    "Lautaro Martínez",
    "Romelu Lukaku",
    "Julian Álvarez",
    "Marcus Rashford",
    "Rafael Leão",
    "Christopher Nkunku",
    "Dušan Vlahović",
    "Khvicha Kvaratskhelia",
    "Darwin Núñez",
    
    # Mediocampistas de Clase Mundial
    "Declan Rice",
    "Aurélien Tchouaméni",
    "Eduardo Camavinga",
    "Martin Ødegaard",
    "İlkay Gündoğan",
    "Frenkie de Jong",
    "Marco Verratti",
    "Nicolo Barella",
    "Jorginho",
    "Mason Mount",
    
    # Defensores y Laterales Top
    "Kyle Walker",
    "Theo Hernández",
    "Reece James",
    "Trent Alexander-Arnold",
    "Achraf Hakimi",
    "João Cancelo",
    "Marquinhos",
    "Kim Min-jae",
    "Eder Militão",
    "Jules Koundé",
    
    # Porteros de Elite
    "Ederson",
    "Marc-André ter Stegen",
    "Mike Maignan",
    "Jan Oblak",
    "Emiliano Martínez",
    "Gianluigi Donnarumma",
    "Edouard Mendy",
    "André Onana",
    
    # Leyendas Históricas Adicionales
    "Ronaldo Nazário",
    "Luís Figo",
    "Paolo Maldini",
    "Franco Baresi",
    "Alessandro Del Piero",
    "Francesco Totti",
    "Roberto Carlos",
    "Cafu",
    "Rivaldo",
    "Roberto Baggio",
    "George Best",
    "Eric Cantona",
    "Ruud van Nistelrooy",
    "Raúl González",
    "Fernando Torres",
    "Didier Drogba",
    "Samuel Eto'o",
    "Carles Puyol"
]

# Directorio donde se guardarán las biografías
OUTPUT_DIR = "biografias_jugadores"

# Configuración de Wikipedia API
USER_AGENT = 'Football-Biography-Scraper/1.0'

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


def search_player(wiki, player_name, max_results=5):
    """
    Busca un jugador en Wikipedia usando búsqueda fuzzy
    
    Args:
        wiki: Objeto Wikipedia API
        player_name: Nombre del jugador a buscar
        max_results: Número máximo de resultados a revisar
    
    Returns:
        Nombre correcto de la página de Wikipedia o None
    """
    try:
        # Intentar primero con el nombre exacto
        page = wiki.page(player_name)
        if page.exists():
            # Verificar que sea futbolista
            summary_lower = page.summary[:500].lower()
            keywords = ['futbolista', 'fútbol', 'football', 'soccer', 'delantero', 
                       'defensa', 'mediocampista', 'portero', 'arquero']
            
            if any(keyword in summary_lower for keyword in keywords):
                return player_name
        
        # Si no funciona, usar búsqueda
        print(f"   → Buscando variaciones de '{player_name}'...")
        
        # Nota: wikipedia-api no tiene método search integrado
        # Alternativa: intentar variaciones comunes
        variations = [
            player_name,
            player_name.replace('í', 'i').replace('é', 'e').replace('ó', 'o'),
            player_name + " (futbolista)",
            player_name.split()[0] + " " + player_name.split()[-1] if len(player_name.split()) > 2 else player_name
        ]
        
        for variation in variations:
            page = wiki.page(variation)
            if page.exists():
                summary_lower = page.summary[:500].lower()
                keywords = ['futbolista', 'fútbol', 'football', 'soccer']
                
                if any(keyword in summary_lower for keyword in keywords):
                    print(f"   ✓ Encontrado como: '{variation}'")
                    return variation
        
        return None
        
    except Exception as e:
        print(f"   ✗ Error buscando '{player_name}': {str(e)}")
        return None


def download_biography(wiki, player_name, output_dir):
    """
    Descarga la biografía de un jugador y la guarda en un archivo .txt
    
    Args:
        wiki: Objeto Wikipedia API
        player_name: Nombre del jugador
        output_dir: Directorio donde guardar el archivo
    
    Returns:
        True si se descargó exitosamente, False en caso contrario
    """
    try:
        # Buscar el jugador
        correct_name = search_player(wiki, player_name)
        
        if not correct_name:
            print(f"❌ No se encontró: {player_name}")
            return False
        
        # Obtener la página
        page = wiki.page(correct_name)
        
        if not page.exists():
            print(f"❌ Página no existe: {player_name}")
            return False
        
        # Extraer información
        title = page.title
        summary = page.summary
        full_text = page.text
        
        # Crear nombre de archivo seguro (sin caracteres especiales)
        safe_filename = "".join(c for c in player_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_filename = safe_filename.replace(' ', '_')
        filepath = os.path.join(output_dir, f"{safe_filename}.txt")
        
        # Guardar en archivo
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"JUGADOR: {title}\n")
            f.write("=" * 80 + "\n\n")
            f.write("RESUMEN:\n")
            f.write("-" * 80 + "\n")
            f.write(summary + "\n\n")
            f.write("BIOGRAFÍA COMPLETA:\n")
            f.write("-" * 80 + "\n")
            f.write(full_text)
        
        print(f"✅ Descargado: {player_name} → {safe_filename}.txt")
        return True
        
    except Exception as e:
        print(f"❌ Error descargando {player_name}: {str(e)}")
        return False


def download_all_biographies(players, output_dir=OUTPUT_DIR, delay=1.0):
    """
    Descarga todas las biografías de la lista de jugadores en español
    
    Args:
        players: Lista de nombres de jugadores
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
        'total': len(players),
        'jugadores_descargados': [],
        'jugadores_fallidos': []
    }
    
    print("=" * 80)
    print(f"DESCARGANDO BIOGRAFÍAS DE {len(players)} JUGADORES")
    print(f"Idioma: ESPAÑOL")
    print(f"Directorio: {output_dir}")
    print("=" * 80 + "\n")
    
    # Descargar cada jugador
    for i, player in enumerate(players, 1):
        print(f"\n[{i}/{len(players)}] Procesando: {player}")
        
        success = download_biography(wiki, player, output_dir)
        
        if success:
            stats['exitosos'] += 1
            stats['jugadores_descargados'].append(player)
        else:
            stats['fallidos'] += 1
            stats['jugadores_fallidos'].append(player)
        
        # Esperar para evitar rate limiting
        if i < len(players):  # No esperar después del último
            time.sleep(delay)
    
    # Imprimir resumen final
    print("\n" + "=" * 80)
    print("RESUMEN DE DESCARGA")
    print("=" * 80)
    print(f"✅ Exitosos: {stats['exitosos']}/{stats['total']}")
    print(f"❌ Fallidos: {stats['fallidos']}/{stats['total']}")
    print(f"📊 Tasa de éxito: {(stats['exitosos']/stats['total']*100):.1f}%")
    
    if stats['jugadores_fallidos']:
        print(f"\n⚠️ Jugadores no encontrados:")
        for player in stats['jugadores_fallidos']:
            print(f"   - {player}")
    
    print(f"\n📁 Archivos guardados en: {output_dir}/")
    print("=" * 80)
    
    return stats


def create_metadata_file(stats, output_dir=OUTPUT_DIR):
    """
    Crea un archivo de metadatos con la información de descarga
    """
    metadata_path = os.path.join(output_dir, "_metadata.txt")
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write("METADATOS DE DESCARGA - BIOGRAFÍAS DE JUGADORES\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total jugadores: {stats['total']}\n")
        f.write(f"Descargados exitosamente: {stats['exitosos']}\n")
        f.write(f"Fallidos: {stats['fallidos']}\n\n")
        
        f.write("JUGADORES DESCARGADOS:\n")
        f.write("-" * 80 + "\n")
        for player in stats['jugadores_descargados']:
            f.write(f"✓ {player}\n")
        
        if stats['jugadores_fallidos']:
            f.write("\nJUGADORES NO ENCONTRADOS:\n")
            f.write("-" * 80 + "\n")
            for player in stats['jugadores_fallidos']:
                f.write(f"✗ {player}\n")
    
    print(f"📝 Metadatos guardados en: {metadata_path}")


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    """
    Ejecutar el script para descargar todas las biografías en español
    """
    
    print("\n🏆 SCRAPER DE BIOGRAFÍAS DE JUGADORES - WIKIPEDIA")
    print("=" * 80 + "\n")
    
    # Confirmar inicio
    input(f"Se descargarán {len(JUGADORES)} biografías en ESPAÑOL. Presiona ENTER para comenzar...")
    
    # Descargar biografías
    stats = download_all_biographies(
        players=JUGADORES,
        output_dir=OUTPUT_DIR,
        delay=1.0  # 1 segundo entre requests
    )
    
    # Crear archivo de metadatos
    create_metadata_file(stats, OUTPUT_DIR)