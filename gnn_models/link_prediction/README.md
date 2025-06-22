🧠 Link Prediction con GNN en Transporte Interurbano

Este proyecto aplica una Graph Neural Network (GNN) para predecir si debería existir una conexión directa (viaje) entre dos puntos del sistema de transporte de buses en Brasil. Utiliza datos reales anonimizados y una arquitectura basada en GCN de PyTorch Geometric.

📁 Estructura del Proyecto

gnntoymodel/
├── data/
│   ├── venda_passagem_01_2020.csv         # Dataset original
│   ├── viajes_limpios.csv                 # Dataset limpio y reducido
│   ├── graph_link_prediction.pt           # Grafo construido con PyG
│   └── label_encoder.pkl                  # Codificador de nodos
├── src/
│   ├── exploratory.ipynb                  # Limpieza y exploración
│   ├── visual.py                          # Visualización inicial
├── gnn_models/
│   ├── link_prediction.py                 # Código principal de entrenamiento y evaluación
│   ├── gcn_model.pt                       # Pesos entrenados (guardado)
├── models/
│   └── gcn_link_model.pt                  # Modelo final entrenado
└── README.md
🎯 Objetivo

Dado un punto de origen y un destino, predecir si existe o debería existir una conexión directa entre ellos.

Esto se plantea como un problema de Link Prediction en grafos dirigidos, usando datos históricos de billetes vendidos.

🧼 1. Limpieza del Dataset

Del archivo original venda_passagem_01_2020.csv, se filtran:

Solo viajes interestaduales y de tipo Regular
Se eliminan duplicados y columnas irrelevantes o con ruido
Se conserva únicamente:
nu_linha
ponto_origem_viagem
ponto_destino_viagem
data_viagem
hora_viagem
El resultado se guarda como viajes_limpios.csv.

🧩 2. Representación como grafo

Nodos: cada localidad (ponto_origem_viagem o ponto_destino_viagem)
Aristas: cada fila del CSV representa un viaje directo (origen → destino)
Features de nodo:
Por defecto, identidad
Posible extensión: codificación de hora o línea
Etiquetas de arista:
1 = conexión real (positiva)
0 = conexión no observada (negativa), muestreada aleatoriamente
El grafo se guarda como graph_link_prediction.pt.

🧠 3. Modelo GNN (GCNEncoder)

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        ...
Dos capas GCN
Devuelve embeddings por nodo
🔄 4. Entrenamiento

Se entrena un GCN para generar embeddings
Se predicen conexiones usando producto punto de embeddings
Se usa pérdida BCEWithLogitsLoss y optimizador Adam
Se equilibran aristas positivas y negativas
for epoch in range(100):
    loss = train()
📊 5. Evaluación

Métricas:
Accuracy
ROC AUC Score
Visualización:
Histogramas de puntuaciones (positivas vs negativas)
Subgrafo con predicciones coloreadas (verde, rojo, azul)
📌 Ejemplo de Visualización del Subgrafo

Nombres reales de nodos (con LabelEncoder)
Colores según tipo de predicción:
✅ Verde: conexión real y predicha como tal
❌ Rojo: conexión real no detectada
❗ Azul: conexión negativa predicha erróneamente como positiva
🧪 Cómo ejecutar

# 1. Activar entorno virtual
source venv-mac/bin/activate

# 2. Ejecutar el pipeline
python gnn_models/link_prediction.py
✅ Requisitos

Python ≥ 3.10
PyTorch ≥ 2.0
PyTorch Geometric
NetworkX
scikit-learn
matplotlib
pandas
Instalación (una vez activado el entorno):

pip install torch torchvision torchaudio
pip install torch-geometric
pip install matplotlib networkx scikit-learn pandas joblib
🧠 Futuras mejoras

Usar codificaciones más ricas (hora, día, tipo de línea)
Agregar pesos a las aristas (frecuencia o volumen)
Usar un grafo temporal
Evaluar con más datos y sobre periodos futuros

