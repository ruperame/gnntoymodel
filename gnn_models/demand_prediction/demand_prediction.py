# demand_prediction.py

import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch.optim import Adam
from torch.nn import MSELoss
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import joblib

# === Preparación ===
os.makedirs("gnn_models/demand_prediction", exist_ok=True)
os.makedirs("../../data", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv("../../data/viajes_limpios.csv")
df["hora"] = pd.to_datetime(df["hora_viagem"], format="%H:%M:%S", errors="coerce").dt.hour
df["dia_semana"] = pd.to_datetime(df["data_viagem"], errors="coerce").dt.dayofweek
df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
df["dia_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
df["dia_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)

df_agg = df.groupby([
    "ponto_origem_viagem", "ponto_destino_viagem", "nu_linha",
    "hora_sin", "hora_cos", "dia_sin", "dia_cos"
]).size().reset_index(name="num_viajes")

# === División por rangos ===
low = df_agg[df_agg["num_viajes"] <= 100]
mid = df_agg[(df_agg["num_viajes"] > 100) & (df_agg["num_viajes"] <= 1000)]
high = df_agg[df_agg["num_viajes"] > 1000]

# === Codificación de nodos ===
nodos = pd.concat([
    low["ponto_origem_viagem"], low["ponto_destino_viagem"],
    mid["ponto_origem_viagem"], mid["ponto_destino_viagem"],
    high["ponto_origem_viagem"], high["ponto_destino_viagem"]
]).unique()
le = LabelEncoder()
le.fit(nodos)
joblib.dump(le, "../../data/label_encoder_demand.pkl")

# === Construcción de grafos ===
def create_data(df_split):
    df_split = df_split.copy()
    df_split["origen_id"] = le.transform(df_split["ponto_origem_viagem"])
    df_split["destino_id"] = le.transform(df_split["ponto_destino_viagem"])
    edge_index = torch.tensor(df_split[["origen_id", "destino_id"]].values.T, dtype=torch.long)
    edge_features = StandardScaler().fit_transform(df_split[["nu_linha", "hora_sin", "hora_cos", "dia_sin", "dia_cos"]])
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    edge_label = torch.tensor(np.log1p(df_split["num_viajes"].values), dtype=torch.float)
    num_nodes = int(edge_index.max()) + 1
    grado = df_split.groupby("origen_id").size().reindex(range(num_nodes), fill_value=0).values.reshape(-1, 1)
    grado_tensor = torch.tensor(StandardScaler().fit_transform(grado), dtype=torch.float)
    return Data(x=grado_tensor, edge_index=edge_index, edge_attr=edge_attr, edge_label=edge_label)

# === Modelo GCN ===
class GCNEdgeRegressor(nn.Module):
    def __init__(self, in_channels_node, in_channels_edge, hidden_channels):
        super().__init__()
        self.gcn1 = GCNConv(in_channels_node, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2 + in_channels_edge, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        z = self.gcn1(x, edge_index)
        z = self.bn1(z)
        z = F.relu(z)
        z = self.gcn2(z, edge_index)
        src, dst = edge_index
        h_src = z[src]
        h_dst = z[dst]
        edge_input = torch.cat([h_src, h_dst, edge_attr], dim=1)
        return self.edge_mlp(edge_input).squeeze()

# === Entrenamiento por franja ===
def run_training(df_split, franja):
    if len(df_split) < 10:
        print(f"[{franja}] No hay suficientes datos para entrenar.")
        return

    df_train, df_test = train_test_split(df_split, test_size=0.2, random_state=42)
    data_train = create_data(df_train)
    data_test = create_data(df_test)

    model = GCNEdgeRegressor(
        in_channels_node=data_train.x.shape[1],
        in_channels_edge=data_train.edge_attr.shape[1],
        hidden_channels=64
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data_train = data_train.to(device)
    data_test = data_test.to(device)

    loss_fn = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    losses = []

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        output = model(data_train.x, data_train.edge_index, data_train.edge_attr)
        loss = loss_fn(output, data_train.edge_label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 10 == 0:
            print(f"[{franja}] Epoch {epoch}, Loss: {loss.item():.4f}")

    # === Evaluación ===
    model.eval()
    preds = model(data_test.x, data_test.edge_index, data_test.edge_attr)
    y_true = np.expm1(data_test.edge_label.cpu().numpy())
    y_pred = np.expm1(preds.detach().cpu().numpy())

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n🔎 [{franja}] Evaluación sobre test:")
    print(f" MSE: {mse:.2f}")
    print(f" R²: {r2:.2f}")
    print(f" MAE: {mae:.2f}")

    # === Gráfico regresión ===
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, edgecolor='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Valor real (nº viajes)")
    plt.ylabel("Valor predicho")
    plt.title(f"Evaluación sobre test ({franja})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../../data/regresion_{franja}.png")
    plt.close()

    # === Curva de pérdida ===
    plt.figure()
    plt.plot(range(1, 101), losses)
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title(f"Curva de pérdida ({franja})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../../data/loss_{franja}.png")
    plt.close()

    # Guardar modelo
    torch.save(model.state_dict(), f'models/gcn_demand_model_{franja}.pt')

# === Ejecutar ===
run_training(low, "baja")
run_training(mid, "media")
run_training(high, "alta")
print(" Modelos entrenados y gráficos generados.")
