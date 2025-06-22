# ¿Qué queremos predecir?
# Si, dado un punto de origen y un destino, existe (o debería existir) una conexión (un viaje directo) entre ambos.

# Estructura del grafo

# Nodos: cada ponto_origem_viagem y ponto_destino_viagem es un nodo.
# Aristas: cada fila representa una arista (un viaje entre origen y destino).
# Features de nodo: de momento, podemos usar:
# codificación one-hot o label encoding de nu_linha
# hora del viaje
# Etiqueta de arista (y):
# 1 si la conexión existe en el grafo (positiva)
# 0 para conexiones negativas (pares origen-destino no observados, muestreados aleatoriamente)

import pandas as pd
import torch
import networkx as nx
import random
import joblib
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from gnn_models.link_prediction.link_prediction import GCNEncoder, get_positive_negative_edges
import os

# Cargar dataset limpio
df = pd.read_csv('../../data/viajes_limpios.csv')  # o el que hayas generado
df = df.drop_duplicates(subset=['ponto_origem_viagem', 'ponto_destino_viagem'])

# Entrenar LabelEncoder con todos los nodos (origen y destino combinados)
nodos = pd.concat([df['ponto_origem_viagem'], df['ponto_destino_viagem']]).unique()
le = LabelEncoder()
le.fit(nodos)

# Codificar los IDs de los nodos
df['origen_id'] = le.transform(df['ponto_origem_viagem'])
df['destino_id'] = le.transform(df['ponto_destino_viagem'])
joblib.dump(le, '../../data/label_encoder.pkl')
# Crear aristas
edge_index = torch.tensor(df[['origen_id', 'destino_id']].values.T, dtype=torch.long)

# Opcional: usar 'hora' o 'nu_linha' como features
x = torch.arange(edge_index.max().item() + 1).unsqueeze(1).float()  # por ahora identidad simple

# Crear objeto Data
data = Data(x=x, edge_index=edge_index)

# Guardar para usar en entrenamiento
torch.save(data, '../../data/graph_link_prediction.pt')

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x  # embeddings finales de los nodos

def get_positive_negative_edges(data, num_negatives=1):
    # Aristas positivas
    pos_edge_index = data.edge_index.T

    # Aristas negativas (no existentes)
    num_nodes = data.num_nodes
    neg_edges = set()
    pos_edges_set = set(map(tuple, pos_edge_index.tolist()))

    while len(neg_edges) < len(pos_edge_index) * num_negatives:
        i = random.randint(0, num_nodes - 1)
        j = random.randint(0, num_nodes - 1)
        if i == j:
            continue
        if (i, j) not in pos_edges_set and (i, j) not in neg_edges:
            neg_edges.add((i, j))

    neg_edge_index = torch.tensor(list(neg_edges), dtype=torch.long).T

    return pos_edge_index.T, neg_edge_index

data = torch.load('../../data/graph_link_prediction.pt', weights_only=False)

pos_edge_index, neg_edge_index = get_positive_negative_edges(data)

print("Aristas positivas:", pos_edge_index.shape[1])
print("Aristas negativas:", neg_edge_index.shape[1])

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

# Definir modelo y optimizador
model = GCNEncoder(in_channels=data.num_node_features, hidden_channels=64)
optimizer = Adam(model.parameters(), lr=0.01)
loss_fn = BCEWithLogitsLoss()

# Mover a GPU si está disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
pos_edge_index = pos_edge_index.to(device)
neg_edge_index = neg_edge_index.to(device)

def train():
    model.train()
    optimizer.zero_grad()

    # 1. Obtener embeddings de los nodos
    z = model(data.x, data.edge_index)

    # 2. Calcular puntuaciones (producto punto) para aristas
    def dot_product(u, v):
        return (z[u] * z[v]).sum(dim=1)

    pos_score = dot_product(pos_edge_index[0], pos_edge_index[1])
    neg_score = dot_product(neg_edge_index[0], neg_edge_index[1])

    # 3. Concatenar puntuaciones y etiquetas
    scores = torch.cat([pos_score, neg_score], dim=0)
    labels = torch.cat([
        torch.ones(pos_score.size(0)),
        torch.zeros(neg_score.size(0))
    ], dim=0).to(device)

    # 4. Calcular pérdida
    loss = loss_fn(scores, labels)
    loss.backward()
    optimizer.step()

    return loss.item()

# Entrenamiento
for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

@torch.no_grad()
def evaluate():
    model.eval()
    z = model(data.x, data.edge_index)

    def dot_product(u, v):
        return (z[u] * z[v]).sum(dim=1)

    pos_score = dot_product(pos_edge_index[0], pos_edge_index[1])
    neg_score = dot_product(neg_edge_index[0], neg_edge_index[1])

    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = np.concatenate([
        np.ones(pos_score.size(0)),
        np.zeros(neg_score.size(0))
    ])

    # Métricas
    auc = roc_auc_score(labels, scores)
    preds = (scores > 0).astype(int)
    acc = accuracy_score(labels, preds)

    print(f"ROC AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")

    # Visualización: histogramas
    plt.hist(pos_score.cpu().numpy(), bins=50, alpha=0.7, label="Positivos", color='green')
    plt.hist(neg_score.cpu().numpy(), bins=50, alpha=0.7, label="Negativos", color='red')
    plt.title("Distribución de scores")
    plt.xlabel("Score (logits)")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Ejecutar evaluación tras entrenamiento
evaluate()

def visualizar_subgrafo(data, z, pos_edge_index, neg_edge_index, num_nodos=100):
    print("Generando subgrafo con predicciones...")
    le = joblib.load('../../data/label_encoder.pkl')
    # Seleccionar nodos al azar
    sampled_nodes = random.sample(range(data.num_nodes), num_nodos)

    # Filtrar aristas positivas y negativas que estén entre esos nodos
    def edge_filter(edge_idx):
        return [
            (int(u), int(v)) for u, v in edge_idx.T.cpu().numpy()
            if u in sampled_nodes and v in sampled_nodes
        ]

    pos_edges = edge_filter(pos_edge_index)
    neg_edges = edge_filter(neg_edge_index)

    # Obtener predicciones (dot product)
    def get_scores(edge_list):
        u = torch.tensor([e[0] for e in edge_list]).to(z.device)
        v = torch.tensor([e[1] for e in edge_list]).to(z.device)
        scores = (z[u] * z[v]).sum(dim=1)
        preds = (scores > 0).cpu().numpy()
        return preds

    pos_preds = get_scores(pos_edges)
    neg_preds = get_scores(neg_edges)

    # Crear grafo en networkx
    G = nx.Graph()
    G.add_nodes_from(sampled_nodes)

    # Añadir aristas con etiquetas de color
    for (u, v), pred in zip(pos_edges, pos_preds):
        G.add_edge(u, v, color='green' if pred == 1 else 'red')

    for (u, v), pred in zip(neg_edges, neg_preds):
        if pred == 1:  # solo dibujamos las negativas predichas como positivas (falsos positivos)
            G.add_edge(u, v, color='blue')

    # Dibujar
    pos = nx.spring_layout(G, seed=42)
    edge_colors = [G[u][v]['color'] for u, v in G.edges]

    # Dibujar con etiquetas
    plt.figure(figsize=(14, 10))
    nx.draw(
        G,
        pos,
        node_size=300,
        with_labels=True,
        labels={n: le.inverse_transform([n])[0] for n in G.nodes},  # mostrar nombre real
        font_size=7,
        edge_color=edge_colors
    )

    # Leyenda de colores
    from matplotlib.lines import Line2D
    leyenda = [
        Line2D([0], [0], color='green', lw=2, label='Real + Predicha (Correcta)'),
        Line2D([0], [0], color='red', lw=2, label='Real + Predicha como 0 (Falso Negativo)'),
        Line2D([0], [0], color='blue', lw=2, label='Negativa predicha como Positiva (Falso Positivo)'),
    ]
    plt.legend(handles=leyenda, loc='best')
    plt.title("Subgrafo con nombres y predicciones")
    plt.tight_layout()
    plt.show()


# Obtener embeddings para visualización
model.eval()
z = model(data.x, data.edge_index)

# Visualizar
visualizar_subgrafo(data, z, pos_edge_index, neg_edge_index)

# Guardar modelo
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/gcn_link_model.pt')
print("✅ Modelo guardado en 'models/gcn_link_model.pt'")

# Mostrar evolución de la pérdida (si has guardado las pérdidas por época)
import matplotlib.pyplot as plt

losses = []
for epoch in range(1, 101):
    loss = train()
    losses.append(loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Visualizar curva de pérdida
plt.plot(range(1, 101), losses)
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.title("Evolución de la pérdida durante el entrenamiento")
plt.grid(True)
plt.tight_layout()
plt.show()

def predict_link(origen_nombre, destino_nombre):
    import torch
    import joblib
    from gnn_models.link_prediction.link_prediction import GCNEncoder
    from torch_geometric.data import Data

    # Cargar grafo y LabelEncoder
    data = torch.load('../../data/graph_link_prediction.pt', weights_only=False)
    le = joblib.load('../../data/label_encoder.pkl')

    # Cargar modelo
    model = GCNEncoder(in_channels=data.num_node_features, hidden_channels=64)
    model.load_state_dict(torch.load('models/gcn_link_model.pt', map_location='cpu'))
    model.eval()

    # Convertir nombres a IDs
    try:
        u = le.transform([origen_nombre])[0]
        v = le.transform([destino_nombre])[0]
    except ValueError:
        print("⚠️ Uno de los nombres no existe en el grafo.")
        return None

    # Predecir
    z = model(data.x, data.edge_index)
    score = (z[u] * z[v]).sum().item()
    prob = torch.sigmoid(torch.tensor(score)).item()

    print(f"🔍 Conexión entre '{origen_nombre}' y '{destino_nombre}':")
    print(f"   Score: {score:.4f} | Probabilidad estimada: {prob:.2%}")
    return prob

# Ejemplo de uso
predict_link("SAO PAULO/SP", "RIO DE JANEIRO/RJ")
