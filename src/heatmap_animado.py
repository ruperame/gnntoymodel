import pandas as pd
import folium
from folium.plugins import TimestampedGeoJson
import json
from datetime import datetime

# Cargar datos
df = pd.read_csv("../data/resultados_franjas.csv")

# Coordenadas por parada (puedes cargarlo de archivo si lo prefieres)
coords = {
    "Acueducto 3": (40.9485273, -4.1177604),
    "Pza. Toros": (40.9421713, -4.1076958),
    "Gerardo Diego": (40.9367195, -4.1055851),
    "Centro": (40.9440341, -4.1227178),
    "Inst. Andrés Laguna": (40.9391215, -4.1180379),
    "Ctra. S. Rafael": (40.9351858, -4.1111758),
    "Estación AVE": (40.9105922, -4.0956776),
}

# Cargar predicciones como si fueran demanda por franja y línea
features = []
for i, row in df.iterrows():
    # solo usamos la estación
    if row["linea"] not in ["Línea 11", "Línea 12"]:
        continue

    parada = "Estación AVE"
    lat, lon = coords[parada]
    timestamp = f"2024-01-01T{row['franja_horaria']}:00"

    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat],
        },
        "properties": {
            "time": timestamp,
            "style": {"color": "red"},
            "icon": "circle",
            "iconstyle": {
                "fillColor": "red",
                "fillOpacity": 0.6,
                "stroke": False,
                "radius": row["prediccion"] / 1.5  # escalado visual
            },
            "popup": f"{row['linea']}<br>Demanda: {row['prediccion']:.1f}<br>Hora: {row['franja_horaria']}"
        }
    })

# Crear mapa base
m = folium.Map(location=[40.93, -4.11], zoom_start=13)

# Crear capa animada
TimestampedGeoJson({
    "type": "FeatureCollection",
    "features": features
}, period="PT2H", add_last_point=True, auto_play=True, loop=True, max_speed=1).add_to(m)

# Guardar mapa
m.save("mapa_heatmap_animado.html")
print("✅ Heatmap animado guardado como 'mapa_heatmap_animado.html'")
