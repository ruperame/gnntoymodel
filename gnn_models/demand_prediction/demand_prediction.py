import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim import Adam
from torch.nn import MSELoss
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import numpy as np

# Crear carpeta si no existe
os.makedirs("gnn_models/demand_prediction", exist_ok=True)

# Paso 1: Cargar dataset limpio y agregar demanda
df = pd.read_csv("../../data/viajes_limpios.csv")

# Convertir hora y día
df["hora"] = pd.to_datetime(df["hora_viagem"], format="%H:%M:%S", errors="coerce").dt.hour
df["dia_semana"] = pd.to_datetime(df["data_viagem"], errors="coerce").dt.dayofweek

# Agrupar por origen, destino, línea, hora y día de la semana
df_agg = df.groupby(
    ["ponto_origem_viagem", "ponto_destino_viagem", "nu_linha", "hora", "dia_semana"]
).size().reset_index(name="num_viajes")

# Eliminar outliers (top 1%)
percentil_99 = df_agg["num_viajes"].quantile(0.99)
df_agg = df_agg[df_agg["num_viajes"] <= percentil_99]

# Separar en train y test
df_train, df_test = train_test_split(df_agg, test_size=0.2, random_state=42)

# Codificar nodos (entrenar con ambos sets combinados para evitar desincronía)
nodos = pd.concat([
    df_train["ponto_origem_viagem"], df_train["ponto_destino_viagem"],
    df_test["ponto_origem_viagem"], df_test["ponto_destino_viagem"]
]).unique()
le = LabelEncoder()
le.fit(nodos)
joblib.dump(le, "../../data/label_encoder_demand.pkl")

# Función para crear objetos Data
def create_data(df_split):
    df_split = df_split.copy()
    df_split["origen_id"] = le.transform(df_split["ponto_origem_viagem"])
    df_split["destino_id"] = le.transform(df_split["ponto_destino_viagem"])

    edge_index = torch.tensor(df_split[["origen_id", "destino_id"]].values.T, dtype=torch.long)
    edge_features = StandardScaler().fit_transform(df_split[["nu_linha", "hora", "dia_semana"]])
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    edge_label = torch.tensor(np.log1p(df_split["num_viajes"].values), dtype=torch.float)
    
    num_nodes = int(edge_index.max()) + 1
    x = torch.arange(num_nodes).unsqueeze(1).float()
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_label=edge_label)

# Crear objetos Data para entrenamiento y test
data_train = create_data(df_train)
data_test = create_data(df_test)

# Guardar
torch.save(data_train, "../../data/graph_demand_train.pt")
torch.save(data_test, "../../data/graph_demand_test.pt")

print("Datos de entrenamiento y test generados.")

# Modelo GCN para regresión sobre aristas
class GCNEdgeRegressor(nn.Module):
    def __init__(self, in_channels_node, in_channels_edge, hidden_channels):
        super().__init__()
        self.gcn1 = GCNConv(in_channels_node, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2 + in_channels_edge, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        z = self.gcn1(x, edge_index)
        z = F.relu(z)
        z = self.gcn2(z, edge_index)
        src, dst = edge_index
        h_src = z[src]
        h_dst = z[dst]
        edge_input = torch.cat([h_src, h_dst, edge_attr], dim=1)
        return self.edge_mlp(edge_input).squeeze()

# Instanciar modelo
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

# Entrenamiento
def train():
    model.train()
    optimizer.zero_grad()
    output = model(data_train.x, data_train.edge_index, data_train.edge_attr)
    loss = loss_fn(output, data_train.edge_label)
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Evaluación
@torch.no_grad()
def evaluate(data_eval, title="Evaluación"):
    model.eval()
    preds = model(data_eval.x, data_eval.edge_index, data_eval.edge_attr)
    y_true = np.expm1(data_eval.edge_label.cpu().numpy())
    y_pred = np.expm1(preds.cpu().numpy())

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"🔎 {title}:")
    print(f"📉 MSE: {mse:.2f}")
    print(f"📈 R²: {r2:.2f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, edgecolor='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Valor real (nº viajes)")
    plt.ylabel("Valor predicho")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Evaluar sobre test
evaluate(data_test, title="Evaluación sobre test")

# Guardar
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/gcn_demand_model.pt')
joblib.dump(le, '../../data/label_encoder.pkl')
print("Modelo guardado en 'models/gcn_demand_model.pt'")
