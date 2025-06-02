import torch
import torch.nn as nn
from torch_geometric.data import Data
from model import GCN
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random

resultados = []

def crear_grafo_linea_11():
    nodos_11 = {
        "Acueducto 3": 0,
        "Pza. Toros": 1,
        "Gerardo Diego": 2,
        "Estación AVE": 3,
    }
    edges_11 = [
        ("Acueducto 3", "Pza. Toros", 4),
        ("Pza. Toros", "Gerardo Diego", 3),
        ("Gerardo Diego", "Estación AVE", 7),
    ]
    edge_index_11 = torch.tensor([[nodos_11[a], nodos_11[b]] for (a, b, _) in edges_11], dtype=torch.long).t()
    edge_attr_11 = torch.tensor([[w] for (_, _, w) in edges_11], dtype=torch.float)

    horaL11 = 0.0 #10:00
    # Parametros: suben, bajan, hora
    x_11 = torch.tensor([
        [5.0, 0.0, horaL11],
        [3.0, 1.0, horaL11],
        [6.0, 0.0, horaL11],
        [0.0, 14.0, horaL11],
    ], dtype=torch.float)

    y_11 = torch.tensor([14.0], dtype=torch.float)
    data_11 = Data(x=x_11, edge_index=edge_index_11, edge_attr=edge_attr_11, y=y_11)
    return data_11, nodos_11

def crear_grafo_linea_12():
    nodos_12 = {
        "Centro": 0,
        "Inst. Andrés Laguna": 1,
        "Ctra. S. Rafael": 2,
        "Estación AVE": 3,
    }
    edges_12 = [
        ("Centro", "Inst. Andrés Laguna", 2),
        ("Inst. Andrés Laguna", "Ctra. S. Rafael", 3),
        ("Ctra. S. Rafael", "Estación AVE", 7),
    ]
    edge_index_12 = torch.tensor([[nodos_12[a], nodos_12[b]] for (a, b, _) in edges_12], dtype=torch.long).t()
    edge_attr_12 = torch.tensor([[w] for (_, _, w) in edges_12], dtype=torch.float)

    horaL12 = 0.0
    x_12 = torch.tensor([
        [8.0, 0.0, horaL12],
        [7.0, 1.0, horaL12],
        [4.0, 2.0, horaL12],
        [0.0, 19.0, horaL12],
    ], dtype=torch.float)

    y_12 = torch.tensor([19.0], dtype=torch.float)
    data_12 = Data(x=x_12, edge_index=edge_index_12, edge_attr=edge_attr_12, y=y_12)
    return data_12, nodos_12

def entrenar_y_visualizar(data, nodos, nombre):
    print(f"\n===== Entrenando modelo para {nombre} =====")
    print(data)
    
    G = nx.DiGraph()
    edge_list = data.edge_index.t().tolist()
    G.add_edges_from(edge_list)
    labels = {v: k for k, v in nodos.items()}
    node_colors = ["gray"] * (len(nodos) - 1) + ["red"]

    plt.figure(figsize=(7, 4))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, labels=labels, node_color=node_colors, node_size=1000, font_size=10)
    plt.title(f"Grafo - {nombre}")
    plt.show()

    model = GCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    estacion_idx = len(nodos) - 1

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[estacion_idx], data.y)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

    print(f"Predicción para la estación: {out[estacion_idx].item():.2f}")
    print(f"Valor real: {data.y.item():.2f}")
    return out[estacion_idx].item()


def visualizar_grafos_combinados(data_11, nodos_11, data_12, nodos_12):
    G = nx.DiGraph()

    edge_list_11 = data_11.edge_index.t().tolist()
    tiempos_11 = [int(attr[0]) for attr in data_11.edge_attr.tolist()]
    G.add_edges_from(edge_list_11)

    offset = max(nodos_11.values()) + 1
    nodos_12_offset = {k: v + offset for k, v in nodos_12.items()}
    # Invertir nodos_12 para mapear índice → nombre
    idx_to_nombre_12 = {v: k for k, v in nodos_12.items()}

    # Aplicar offset a cada índice según nombre
    edge_list_12 = [
        [nodos_12_offset[idx_to_nombre_12[a]], nodos_12_offset[idx_to_nombre_12[b]]]
        for [a, b] in data_12.edge_index.t().tolist()
    ]

    tiempos_12 = [int(attr[0]) for attr in data_12.edge_attr.tolist()]
    G.add_edges_from(edge_list_12)

    labels = {**{v: k for k, v in nodos_11.items()},
              **{v + offset: k for k, v in nodos_12.items()}}
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=800)
    nx.draw_networkx_labels(G, pos, labels, font_size=9)

    nx.draw_networkx_edges(G, pos, edgelist=edge_list_11, edge_color='blue', width=2, arrows=True)
    nx.draw_networkx_edges(G, pos, edgelist=edge_list_12, edge_color='green', width=2, arrows=True)

    edge_labels_11 = {(a, b): f"{t} min" for (a, b), t in zip(edge_list_11, tiempos_11)}
    edge_labels_12 = {(a, b): f"{t} min" for (a, b), t in zip(edge_list_12, tiempos_12)}
    edge_labels = {**edge_labels_11, **edge_labels_12}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Líneas 11 (azul) y 12 (verde) con tiempos de recorrido")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# === MAIN EXECUTION ===
def generar_grafo_por_hora(hora_id, hora_texto):
    print(f"\n===== ⏰ Franja horaria: {hora_texto} (código {hora_id}) =====")

    # Línea 11
    nodos_11 = {
        "Acueducto 3": 0,
        "Pza. Toros": 1,
        "Gerardo Diego": 2,
        "Estación AVE": 3,
    }
    edges_11 = [
        ("Acueducto 3", "Pza. Toros", 4),
        ("Pza. Toros", "Gerardo Diego", 3),
        ("Gerardo Diego", "Estación AVE", 7),
    ]
    edge_index_11 = torch.tensor([[nodos_11[a], nodos_11[b]] for (a, b, _) in edges_11], dtype=torch.long).t()
    edge_attr_11 = torch.tensor([[w] for (_, _, w) in edges_11], dtype=torch.float)

    demanda_hora_11 = {
    0.0: [6, 4, 5, 15],   # 10:00
    1.0: [3, 2, 3, 8],    # 12:00
    2.0: [7, 5, 6, 18],   # 16:00
    3.0: [2, 1, 2, 6],    # 20:00
    }
    suben_bajan_11 = demanda_hora_11[hora_id]
    x_11 = torch.tensor([
        [suben_bajan_11[0], random.uniform(0, 2), hora_id],
        [suben_bajan_11[1], random.uniform(0, 2), hora_id],
        [suben_bajan_11[2], random.uniform(0, 2), hora_id],
        [0.0, suben_bajan_11[3], hora_id],
    ], dtype=torch.float)
    y_11 = torch.tensor([suben_bajan_11[3]], dtype=torch.float)  # target: estación
    data_11 = Data(x=x_11, edge_index=edge_index_11, edge_attr=edge_attr_11, y=y_11)

    # Línea 12
    nodos_12 = {
        "Centro": 0,
        "Inst. Andrés Laguna": 1,
        "Ctra. S. Rafael": 2,
        "Estación AVE": 3,
    }
    edges_12 = [
        ("Centro", "Inst. Andrés Laguna", 2),
        ("Inst. Andrés Laguna", "Ctra. S. Rafael", 3),
        ("Ctra. S. Rafael", "Estación AVE", 7),
    ]
    edge_index_12 = torch.tensor([[nodos_12[a], nodos_12[b]] for (a, b, _) in edges_12], dtype=torch.long).t()
    edge_attr_12 = torch.tensor([[w] for (_, _, w) in edges_12], dtype=torch.float)

    demanda_hora_12 = {
    0.0: [9, 6, 5, 20],
    1.0: [4, 3, 2, 9],
    2.0: [10, 7, 6, 23],
    3.0: [3, 2, 1, 6],
    }
    suben_bajan_12 = demanda_hora_12[hora_id]
    x_12 = torch.tensor([
        [suben_bajan_12[0], random.uniform(0, 2), hora_id],
        [suben_bajan_12[1], random.uniform(0, 2), hora_id],
        [suben_bajan_12[2], random.uniform(0, 2), hora_id],
        [0.0, suben_bajan_12[3], hora_id],
    ], dtype=torch.float)
    y_12 = torch.tensor([suben_bajan_12[3]], dtype=torch.float)

    data_12 = Data(x=x_12, edge_index=edge_index_12, edge_attr=edge_attr_12, y=y_12)

    # Entrenamiento y recolección de resultados
    pred_11 = entrenar_y_visualizar(data_11, nodos_11, f"Línea 11 ({hora_texto})")
    resultados.append({"franja_horaria": hora_texto, "linea": "Línea 11", "prediccion": pred_11, "real": y_11.item()})

    pred_12 = entrenar_y_visualizar(data_12, nodos_12, f"Línea 12 ({hora_texto})")
    resultados.append({"franja_horaria": hora_texto, "linea": "Línea 12", "prediccion": pred_12, "real": y_12.item()})



# Franjas horarias
franjas = [
    (0.0, "10:00"),
    (1.0, "12:00"),
    (2.0, "16:00"),
    (3.0, "20:00"),
]

for hora_id, hora_texto in franjas:
    generar_grafo_por_hora(hora_id, hora_texto)

 # Guardar resultados en CSV
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("../data/resultados_franjas.csv", index=False)
print("✅ Resultados guardados en data/resultados_franjas.csv")



