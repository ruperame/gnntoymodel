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

# link_prediction.py - Versión mejorada con features enriquecidos, GCN mejorado y muestreo negativo refinado

import os
import pandas as pd
import torch
import random
import joblib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

# === Paso 1: Cargar y procesar datos ===
df = pd.read_csv('../../data/viajes_limpios.csv')
df = df.drop_duplicates(subset=['ponto_origem_viagem', 'ponto_destino_viagem'])
df['hora'] = pd.to_datetime(df['hora_viagem'], format="%H:%M:%S", errors="coerce").dt.hour
df['ponto_origem_viagem'] = df['ponto_origem_viagem'].str.strip()
df['ponto_destino_viagem'] = df['ponto_destino_viagem'].str.strip()

# Crear codificador de nodos
nodos = pd.concat([df['ponto_origem_viagem'], df['ponto_destino_viagem']]).unique()
le = LabelEncoder()
le.fit(nodos)
df['origen_id'] = le.transform(df['ponto_origem_viagem'])
df['destino_id'] = le.transform(df['ponto_destino_viagem'])
joblib.dump(le, '../../data/label_encoder.pkl')

# === Paso 2: Crear features enriquecidos ===
df_nodes = pd.DataFrame({'node': nodos})
df_nodes['estado'] = df_nodes['node'].str[-3:]
df_nodes['ciudad'] = df_nodes['node'].str[:-3].str.strip()

hora_media = df.groupby('ponto_origem_viagem')['hora'].mean().reset_index()
lineas = df.groupby('ponto_origem_viagem')['nu_linha'].agg(lambda x: x.mode()[0]).reset_index()
df_nodes = df_nodes.merge(hora_media, left_on='node', right_on='ponto_origem_viagem', how='left')
df_nodes = df_nodes.merge(lineas, left_on='node', right_on='ponto_origem_viagem', how='left')
df_nodes = df_nodes.fillna(0)

estado_ohe = pd.get_dummies(df_nodes['estado'])
x_features = pd.concat([df_nodes[['hora']], estado_ohe], axis=1)
scaler = StandardScaler()
x_tensor = torch.tensor(scaler.fit_transform(x_features), dtype=torch.float)

edge_index = torch.tensor(df[['origen_id', 'destino_id']].values.T, dtype=torch.long)
data = Data(x=x_tensor, edge_index=edge_index)
torch.save(data, '../../data/graph_link_prediction.pt')

# === Paso 3: GCN mejorado ===
import torch.nn as nn
import torch.nn.functional as F

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(0.2)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

# === Paso 4: Negativos inteligentes ===
def get_positive_negative_edges(data, df, num_negatives=1):
    pos_edge_index = data.edge_index.T
    num_nodes = data.num_nodes
    neg_edges = set()
    pos_edges_set = set(map(tuple, pos_edge_index.tolist()))

    for o in df['origen_id'].unique():
        destinos_pos = set(df[df['origen_id'] == o]['destino_id'])
        for _ in range(num_negatives):
            d = random.choice(list(set(range(num_nodes)) - destinos_pos - {o}))
            if (o, d) not in pos_edges_set:
                neg_edges.add((o, d))

    neg_edge_index = torch.tensor(list(neg_edges), dtype=torch.long).T
    return pos_edge_index.T, neg_edge_index

# === Paso 5: Entrenamiento ===
data = torch.load('../../data/graph_link_prediction.pt', weights_only=False)
pos_edge_index, neg_edge_index = get_positive_negative_edges(data, df)

model = GCNEncoder(in_channels=data.num_node_features, hidden_channels=64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
pos_edge_index = pos_edge_index.to(device)
neg_edge_index = neg_edge_index.to(device)
optimizer = Adam(model.parameters(), lr=0.01)
loss_fn = BCEWithLogitsLoss()

losses = []
def train():
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.edge_index)
    def dot_product(u, v): return (z[u] * z[v]).sum(dim=1)
    pos_score = dot_product(pos_edge_index[0], pos_edge_index[1])
    neg_score = dot_product(neg_edge_index[0], neg_edge_index[1])
    scores = torch.cat([pos_score, neg_score], dim=0)
    labels = torch.cat([
        torch.ones(pos_score.size(0)),
        torch.zeros(neg_score.size(0))
    ], dim=0).to(device)
    loss = loss_fn(scores, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(1, 101):
    loss = train()
    losses.append(loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# === Paso 6: Evaluación ===
@torch.no_grad()
def evaluate():
    model.eval()
    z = model(data.x, data.edge_index)
    def dot_product(u, v): return (z[u] * z[v]).sum(dim=1)
    pos_score = dot_product(pos_edge_index[0], pos_edge_index[1])
    neg_score = dot_product(neg_edge_index[0], neg_edge_index[1])
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = np.concatenate([
        np.ones(pos_score.size(0)),
        np.zeros(neg_score.size(0))
    ])
    auc = roc_auc_score(labels, scores)
    preds = (scores > 0).astype(int)
    acc = accuracy_score(labels, preds)
    print(f"ROC AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    plt.hist(pos_score.cpu().numpy(), bins=50, alpha=0.7, label="Positivos", color='green')
    plt.hist(neg_score.cpu().numpy(), bins=50, alpha=0.7, label="Negativos", color='red')
    plt.legend()
    plt.title("Distribución de scores")
    plt.show()

evaluate()

# === Paso 7: Visualizar curva de pérdida ===
plt.plot(range(1, 101), losses)
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.title("Evolución de la pérdida durante el entrenamiento")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Paso 8: Guardar modelo y función predictiva ===
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/gcn_link_model.pt')
print("✅ Modelo guardado en 'models/gcn_link_model.pt'")

# === Predicción individual ===
def predict_link(origen_nombre, destino_nombre):
    data = torch.load('../../data/graph_link_prediction.pt', weights_only=False)
    le = joblib.load('../../data/label_encoder.pkl')
    model = GCNEncoder(in_channels=data.num_node_features, hidden_channels=64)
    model.load_state_dict(torch.load('models/gcn_link_model.pt', map_location='cpu'))
    model.eval()
    try:
        u = le.transform([origen_nombre])[0]
        v = le.transform([destino_nombre])[0]
    except ValueError:
        print("⚠️ Uno de los nombres no existe en el grafo.")
        return None
    z = model(data.x, data.edge_index)
    score = (z[u] * z[v]).sum().item()
    prob = torch.sigmoid(torch.tensor(score)).item()
    print(f"🔍 Conexión entre '{origen_nombre}' y '{destino_nombre}':\n   Score: {score:.4f} | Probabilidad estimada: {prob:.2%}")
    return prob