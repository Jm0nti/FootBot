import pandas as pd
from num2words import num2words

# Cargar los csv
players = pd.read_csv(r"C:\Users\juanl\Documents\SistemaMultiagente_PLN\exploratorio\players.csv")
valuations = pd.read_csv(r"C:\Users\juanl\Documents\SistemaMultiagente_PLN\exploratorio\player_valuations.csv")


# Convertir fecha a tipo datetime
valuations["date"] = pd.to_datetime(valuations["date"])

# Obtener la valoración más reciente por jugador
latest_val = valuations.sort_values("date").groupby("player_id").tail(1)

# Unir con los nombres de los jugadores
merged = latest_val.merge(players, on="player_id", how="left")

# ---- NUEVA FUNCIÓN ----
def format_value_parts(euros):
    """
    Retorna (numero_formateado, numero_en_palabras, unidad)
    Ej: (3, 'tres', 'millones de euros')
    """
    if euros >= 1_000_000:
        valor = euros / 1_000_000
        valor_red = int(valor) if valor.is_integer() else round(valor, 2)
        unidad = "millones de euros"
        numero_palabras = num2words(valor_red, lang="es")
        return valor_red, numero_palabras, unidad

    elif euros >= 1_000:
        valor = euros / 1_000
        valor_red = int(valor) if valor.is_integer() else round(valor, 2)
        unidad = "mil euros"
        numero_palabras = num2words(valor_red, lang="es")
        return valor_red, numero_palabras, unidad

    else:
        valor_red = euros
        unidad = "euros"
        numero_palabras = num2words(valor_red, lang="es")
        return valor_red, numero_palabras, unidad


# Crear la frase final
def build_description(row):
    valor, valor_palabras, unidad = format_value_parts(row["market_value_in_eur_x"])
    return (
        f"El jugador {row['name']} está valorado en "
        f"{valor} ({valor_palabras}) {unidad}."
    )

merged["descripcion"] = merged.apply(build_description, axis=1)

# Exportar CSV
merged[["descripcion"]].to_csv("valoraciones_finales.csv", index=False)

print("CSV generado exitosamente.")