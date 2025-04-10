import torch
import torch.nn as nn
from torch_geometric.data import Data
from model import GCN
import networkx as nx
import matplotlib.pyplot as plt

# Definir nodos y su índice
nodes = {
    "Acueducto 3": 0,
    "Pza. Toros": 1,
    "Gerardo Diego": 2,
    "Centro": 3,
    "Ctra. S. Rafael": 4,
    "Inst. Andrés Laguna": 5,
    "Estación AVE": 6,
}

# Definir aristas (ida hacia estación)
edges = [
    ("Acueducto 3", "Pza. Toros"),
    ("Pza. Toros", "Gerardo Diego"),
    ("Gerardo Diego", "Estación AVE"),
    ("Centro", "Ctra. S. Rafael"),
    ("Ctra. S. Rafael", "Inst. Andrés Laguna"),
    ("Inst. Andrés Laguna", "Estación AVE"),
]

# Convertimos a índices
edge_index = torch.tensor([[nodes[a], nodes[b]] for (a, b) in edges], dtype=torch.long).t()

# Atributos de nodos: [suben, bajan]
x = torch.tensor([
    [5.0, 0.0],    # Acueducto 3
    [3.0, 1.0],    # Pza. Toros
    [6.0, 0.0],    # Gerardo Diego
    [8.0, 0.0],    # Centro
    [4.0, 2.0],    # Ctra. S. Rafael
    [7.0, 1.0],    # Inst. Andrés Laguna
    [0.0, 22.0],   # Estación AVE
], dtype=torch.float)

# Etiqueta: total de pasajeros que llegan a la estación (suma de todos los que suben)
y = torch.tensor([33.0])  # 5+3+6+8+4+7

# Crear grafo
data = Data(x=x, edge_index=edge_index, y=y)

print(data)
print("Nodos:", data.num_nodes)
print("Aristas:", data.num_edges)
print("Features por nodo:", data.num_node_features)

# Visualización con NetworkX
G = nx.DiGraph()
edge_list = edge_index.t().tolist()
G.add_edges_from(edge_list)

# Etiquetas de los nodos
labels = {v: k for k, v in nodes.items()}
node_colors = ["gray"] * 6 + ["red"]

plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, labels=labels, node_color=node_colors, node_size=1000, font_size=10)
plt.title("Líneas 11 y 12 hacia Estación AVE")
plt.show()

# Entrenamiento del modelo
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[6], data.y)  # nodo Estación
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

print(f"\nPredicción para la estación: {out[6].item():.2f}")
print(f"Valor real: {data.y.item():.2f}")
