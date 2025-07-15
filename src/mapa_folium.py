import pandas as pd
import folium

df = pd.read_csv("../data/ubicaciones_paradas.csv")

# Diccionario de coordenadas por parada
coords = {row["parada"]: (row["latitud"], row["longitud"]) for _, row in df.iterrows()}

# Crear mapa centrado en Segovia
m = folium.Map(location=[40.940, -4.11], zoom_start=13)

# Definir recorridos
linea_11 = ["Acueducto 3", "Plaza de Toros", "Gerardo Diego", "Estación AVE"]
linea_12 = ["Centro", "Instituto Andrés Laguna", "Carretera San Rafael", "Estación AVE"]

# Tiempos de recorrido
tiempos_11 = {
    ("Acueducto 3", "Plaza de Toros"): 4,
    ("Plaza de Toros", "Gerardo Diego"): 3,
    ("Gerardo Diego", "Estación AVE"): 7,
}

tiempos_12 = {
    ("Centro", "Instituto Andrés Laguna"): 2,
    ("Instituto Andrés Laguna", "Carretera San Rafael"): 3,
    ("Carretera San Rafael", "Estación AVE"): 7,
}

# Dibujar línea 11 (azul), tramo a tramo con tooltip
for (origen, destino), tiempo in tiempos_11.items():
    folium.PolyLine(
        locations=[coords[origen], coords[destino]],
        color="blue",
        weight=4,
        tooltip=f"{origen} → {destino}: {tiempo} min"
    ).add_to(m)

# Dibujar línea 12 (verde), tramo a tramo con tooltip
for (origen, destino), tiempo in tiempos_12.items():
    folium.PolyLine(
        locations=[coords[origen], coords[destino]],
        color="green",
        weight=4,
        tooltip=f"{origen} → {destino}: {tiempo} min"
    ).add_to(m)

# Marcar paradas
for parada, (lat, lon) in coords.items():
    folium.CircleMarker(
        location=(lat, lon),
        radius=5,
        popup=parada,
        color="black",
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

# Guardar mapa
m.save("mapa_lineas_11_12.html")
print("Mapa generado como 'mapa_lineas_11_12.html'")
