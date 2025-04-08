import torch
from torch_geometric.data import Data
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from model import GCN



# Nodos: 6 (P1 a Estación)
# Aristas: de P1 a P2, ..., P5 a Estación
edge_index = torch.tensor([
    [0, 1, 2, 3, 4],  # De
    [1, 2, 3, 4, 5]   # A
], dtype=torch.long)

# Atributos de los nodos: cantidad de pasajeros que suben en cada parada
# Ejemplo: en P1 suben 3 personas, en P2 suben 10, etc.
x = torch.tensor([
    [3.0],   # P1
    [10.0],  # P2
    [5.0],   # P3
    [8.0],   # P4
    [2.0],   # P5
    [0.0],   # Estación (valor a predecir)
], dtype=torch.float)

# Etiqueta: número de pasajeros que llegarán a la estación
# Para simplificar, suponemos que es la suma total:
y = torch.tensor([28.0])  

# Crear objeto Data
data = Data(x=x, edge_index=edge_index, y=y)

#Para que las conexiones sean bidireccionales
#edge_index = torch.tensor([
#    [0, 1, 2, 3, 4, 1, 2, 3, 4, 5],
#    [1, 2, 3, 4, 5, 0, 1, 2, 3, 4]
#], dtype=torch.long)

print(data)
print("Nodos:", data.num_nodes)
print("Aristas:", data.num_edges)
print("Features por nodo:", data.num_node_features)

import networkx as nx
import matplotlib.pyplot as plt

# Crear grafo de NetworkX desde edge_index
G = nx.Graph()
edge_list = edge_index.t().tolist()
G.add_edges_from(edge_list)

# Etiquetas de los nodos para graficar
labels = {
    0: "P1",
    1: "P2",
    2: "P3",
    3: "P4",
    4: "P5",
    5: "Estación"
}

# Colores: gris para paradas, rojo para la estación
node_colors = ["gray"] * 5 + ["red"]

# Dibujar grafo
plt.figure(figsize=(8, 4))
pos = nx.spring_layout(G, seed=42)  # Posiciones
nx.draw(G, pos, with_labels=True, labels=labels,
        node_color=node_colors, node_size=800, font_size=12)
plt.title("Grafo de la Línea de Autobús")
plt.show()


model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[5], data.y)  #estación
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")


print(f"Predicción para la estación: {out[5].item():.2f}")
print(f"Valor real: {data.y.item():.2f}")