from PIL import Image, ImageDraw
import io
import base64
from .image_store import store_image_bytes
from langchain.tools import tool
import os
import json



def draw_marker(base_img: Image.Image, x: int, y: int, marker_path: str = "app/recursos/Marcadores/Rojo.png", size: int = 40) -> None:
    """
    Dibuja un marcador SVG en la posición (x, y) sobre la imagen base.
    Permite recolorear el SVG si se especifica color.

    Parámetros:
    - base_img: objeto PIL.Image sobre el cual se dibuja
    - x, y: coordenadas del centro del marcador
    - marker_path: path para png del marcador
    - size: tamaño (en píxeles) al que escalar el SVG
    """

    # No se especifica una opción para seleccionar color porque implica transformar de SVG a PNG 
    # complicando la función y las dependencias necesarias, por simplicidad se utiliza un PNG directamente


    marker_img = Image.open(marker_path).convert("RGBA")

    # Escalar el marcador al tamaño deseado
    marker_img = marker_img.resize((size, size), Image.LANCZOS)

    # Calcular posición para centrar el marcador
    w, h = marker_img.size
    position = (x - w // 2, y - h) # La base del marcador toca (x, y)

    # Pegar con transparencia
    base_img.paste(marker_img, position, marker_img)


def mark_locations_on_map(map_image_path: str, locations: list[tuple[int, int]], output_path: str, size: int  = 50) -> None:
    """
    Marca una ubicación específica en un mapa y guarda la imagen resultante.

    Parámetros:
    - map_image_path: ruta a la imagen del mapa
    - location: tupla (x, y) con las coordenadas donde marcar
    - output_path: ruta para guardar la imagen marcada
    - color: color del marcador
    """
    # Cargar la imagen del mapa
    map_img = Image.open(map_image_path)

    # Dibujar el marcador en la ubicación especificada
    for location in locations:
        draw_marker(map_img, location[0], location[1], size= size)

    # Guardar la imagen resultante
    map_img.save(output_path)


def read_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data

import os
import json

def mark_location_llm(ubicaciones: str, campus: str) -> str:
    """
    Marca una o varias ubicaciones en el mapa del campus correspondiente.

    Parámetros:
    - ubicaciones: nombre o conjunto de ubicaciones a marcar (ejemplo: "12", "M7", "papelerias", "Caf. 4", etc.)
    - campus: nombre del campus donde se encuentran los lugares ("Volador" o "Robledo")

    Retorna:
    - Diccionario JSON con mensaje, status y otros datos si aplica.
    """
    result = {
        "status": 0,
        "mensaje": "Error desconocido.",
        "path": None
    }

    try:
        valid, result, ubicaciones, campus, _ = validate_location(ubicaciones, campus)

        if not valid:
            return json.dumps(result, ensure_ascii = False)

        # --- Generar bytes de imagen en memoria (no escribir a disco) ---
        img_bytes, file_name, mime = generate_marked_image_bytes(ubicaciones, campus)

        # Codificar la imagen como base64 para enviarla como "blob" en JSON
        img_b64 = base64.b64encode(img_bytes).decode('ascii')

        # Éxito — devolvemos blob (base64), mime y nombre de archivo
        result.update({
            "status": 1,
            "file_name": file_name,
            "mime": mime,
            "blob_base64": img_b64,
            "mensaje": f"He marcado {ubicaciones} en el mapa del campus {campus}."
        })

    except Exception as e:
        result["mensaje"] = f"Ocurrió un error inesperado al marcar la ubicación: {str(e)}"
        result["status"] = 0

    return json.dumps(result, ensure_ascii = False)


def generate_marked_image_bytes(ubicaciones: str, campus: str) -> tuple[bytes, str, str]:
    """
    Genera una imagen con las ubicaciones marcadas y devuelve los bytes, el nombre de archivo y el mime type.

    Retorna:
      (img_bytes, file_name, mime)
    Lanza FileNotFoundError si no encuentra recursos necesarios, ValueError para errores de entrada.
    """
    valid, result, ubicaciones, campus, coordenadas = validate_location(ubicaciones, campus)

    if not valid:
        raise ValueError(f"{result}")
    
    campus_dict = {
        "volador": "MapaVolador.jpg",
        "robledo": "MapaRobledo.jpg"
    }
    img_campus = campus_dict.get(campus)
    path_imagenes = "app/recursos"

    if not img_campus or not os.path.exists(f"{path_imagenes}/{img_campus}"):
        raise FileNotFoundError(f"No se encontró la imagen del campus '{campus}'.")

    # Cargar la imagen del campus
    map_img = Image.open(f"{path_imagenes}/{img_campus}").convert("RGBA")

    # Dibujar los marcadores sobre la copia
    for location in coordenadas:
        draw_marker(map_img, location[0], location[1], size=90)

    # Guardar la imagen en un buffer en formato JPEG
    buffer = io.BytesIO()
    rgb_img = map_img.convert("RGB")
    rgb_img.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    img_bytes = buffer.read()
    file_name = f"Bloque_marcado_{ubicaciones}.jpg"
    mime = "image/jpeg"

    return img_bytes, file_name, mime

def validate_location(ubicaciones: str, campus: str) -> tuple[bool, dict]:
    result = {
        "status": 1,
        "mensaje": None,
        "path": None
    }
    valid = True
    try:
        # --- Validación inicial ---
        if not ubicaciones:
            result["mensaje"] = "Error: se debe proporcionar al menos una ubicación."
            return False, result, result, ubicaciones, campus, coordenadas

        # --- Cargar datos de punteros ---
        try:
            datos_sedes = read_json("app/recursos/punteros.json")
        except FileNotFoundError:
            result["mensaje"] = "Error: no se encontró el archivo 'recursos/punteros.json'."
            return False, result, ubicaciones, campus, coordenadas
        except Exception as e:
            result["mensaje"] = f"Error al leer los datos de ubicaciones: {e}"
            return False, result, ubicaciones, campus, coordenadas

        # --- Filtrar campus ---
        campus = campus.strip().lower()
        bloques_campus = {
            key: value for k, v in datos_sedes.items()
            if isinstance(v, dict) and campus in k.lower()
            for key, value in v.items()
        }

        if not bloques_campus:
            result["mensaje"] = f"Error: no se encontraron datos para el campus '{campus}'."
            return False, result, ubicaciones, campus, coordenadas

        # --- Buscar ubicación ---
        ubicaciones = ubicaciones.strip()
        coordenadas = bloques_campus.get(ubicaciones)

        if not coordenadas:
            result["mensaje"] = (
                f"No se encontró la ubicación '{ubicaciones}' en el campus {campus}. "
                "Verifica el nombre."
            )
            return False, result, ubicaciones, campus, coordenadas

        # --- Determinar imagen ---
        campus_dict = {
            "volador": "MapaVolador.jpg",
            "robledo": "MapaRobledo.jpg"
        }
        img_campus = campus_dict.get(campus)
        path_imagenes = "app/recursos"

        if not img_campus or not os.path.exists(f"{path_imagenes}/{img_campus}"):
            result["mensaje"] = f"Error: no se encontró la imagen del campus '{campus}'."
            return False, result, ubicaciones, campus, coordenadas
        
    except Exception as e:
        result["mensaje"] = f"Ocurrió un error inesperado al marcar la ubicación: {str(e)}"
        result["status"] = 0
        valid = False

    return valid, result, ubicaciones, campus, coordenadas

def generate_and_store_marked_image(ubicaciones: str, campus: str, ttl: int = 300) -> dict:
    """Genera la imagen marcada, la almacena en el store en memoria y devuelve metadata con image_id.

    Retorna json: {image_id, file_name, mime}
    Lanza excepciones si algo falla.
    """
    try:
        img_bytes, file_name, mime = generate_marked_image_bytes(ubicaciones, campus)
    except Exception as e:
        return json.dumps({"status": 0, "mensaje": f"Ha ocurrido un error: {e}"}, ensure_ascii = False)
    image_id = store_image_bytes(img_bytes, file_name, mime, ttl=ttl)
    result =  {
        "status": 1,
        "image_id": image_id, 
        "file_name": file_name, 
        "mime": mime,
        "mensaje": f"He marcado {ubicaciones} en el mapa del campus {campus}."}

    # Las respuestas de las funciones @tool deben ser siempre de tipo str, por lo tanto se deben 
    # guardar en formato json para ser luego extraido
    return json.dumps(result, ensure_ascii = False)


@tool("mark_location_volador", return_direct=True)
def mark_location_volador(ubicaciones: str) -> str:
    """
    Marca una ubicación en el mapa del campus Volador.
    Parámetros:
    - ubicaciones: nombre o conjunto de ubicaciones a marcar (ejemplo: "12",  "Caf. 4", etc.)
    """
    # return mark_location_llm(ubicaciones, "Volador")
    return generate_and_store_marked_image(ubicaciones, "Volador")

@tool("mark_location_robledo", return_direct=True)
def mark_location_robledo(ubicaciones: str) -> str:
    """
    Marca una ubicación en el mapa del campus Robledo.
    Parámetros:
    - ubicaciones: nombre o conjunto de ubicaciones a marcar (ejemplo: "M7", "papelerias", etc.)
    """
    # return mark_location_llm(ubicaciones, "Robledo")
    return generate_and_store_marked_image(ubicaciones, "Robledo")


def json_response(source, data, success=True):
    """
    Retorna una respuesta en formato de diccionario.
    Si 'data' es un string con un JSON válido, lo decodifica automáticamente.
    """
    # Intentar decodificar si 'data' es una cadena JSON
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            # No era JSON válido, se deja como texto
            pass

    return {
        "source": source,
        "data": data,
        "success": success
    }
