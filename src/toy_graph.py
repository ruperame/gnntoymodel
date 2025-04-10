import torch
import torch.nn as nn
from torch_geometric.data import Data
from model import GCN
import networkx as nx
import matplotlib.pyplot as plt

def crear_grafo_linea_11():
    nodos_11 = {
        "Acueducto 3": 0,
        "Pza. Toros": 1,
        "Gerardo Diego": 2,
        "Estación AVE": 3,
    }
    edges_11 = [
        ("Acueducto 3", "Pza. Toros"),
        ("Pza. Toros", "Gerardo Diego"),
        ("Gerardo Diego", "Estación AVE"),
    ]
    edge_index_11 = torch.tensor([[nodos_11[a], nodos_11[b]] for (a, b) in edges_11], dtype=torch.long).t()

    x_11 = torch.tensor([
        [5.0, 0.0],  # Acueducto 3
        [3.0, 1.0],  # Pza. Toros
        [6.0, 0.0],  # Gerardo Diego
        [0.0, 14.0], # Estación
    ], dtype=torch.float)

    y_11 = torch.tensor([14.0])
    data_11 = Data(x=x_11, edge_index=edge_index_11, y=y_11)
    return data_11, nodos_11

def crear_grafo_linea_12():
    nodos_12 = {
        "Centro": 0,
        "Ctra. S. Rafael": 1,
        "Inst. Andrés Laguna": 2,
        "Estación AVE": 3,
    }
    edges_12 = [
        ("Centro", "Ctra. S. Rafael"),
        ("Inst. Andrés Laguna", "Ctra. S. Rafael"),
        ("Ctra. S. Rafael", "Estación AVE"),
    ]
    edge_index_12 = torch.tensor([[nodos_12[a], nodos_12[b]] for (a, b) in edges_12], dtype=torch.long).t()

    x_12 = torch.tensor([
        [8.0, 0.0],  # Centro
        [4.0, 2.0],  # Ctra. S. Rafael
        [7.0, 1.0],  # Inst. Andrés Laguna
        [0.0, 19.0], # Estación
    ], dtype=torch.float)

    y_12 = torch.tensor([19.0])
    data_12 = Data(x=x_12, edge_index=edge_index_12, y=y_12)
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


# Línea 11
data_11, nodos_11 = crear_grafo_linea_11()
entrenar_y_visualizar(data_11, nodos_11, "Línea 11")

# Línea 12
data_12, nodos_12 = crear_grafo_linea_12()
entrenar_y_visualizar(data_12, nodos_12, "Línea 12")
