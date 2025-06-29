# README - Link Prediction Mejorado

Este archivo describe los cambios introducidos respecto a la versión anterior del modelo de predicción de enlaces.

---

## ✨ Mejoras implementadas

### 1. **Features enriquecidos en los nodos**

* Se han introducido nuevas variables para cada nodo:

  * Hora media de salida desde el nodo.
  * Estado extraído del nombre del nodo.
  * Línea más frecuente asociada al nodo.
* Se realiza one-hot encoding del estado y normalización estándar de las features.

### 2. **Modelo GCN mejorado**

* Se añaden:

  * `BatchNorm1d` para estabilizar el entrenamiento.
  * `Dropout` para evitar sobreajuste.
  * Arquitectura de 2 capas GCN.

### 3. **Generación de negativos refinada**

* Los pares negativos (no conectados) se seleccionan de forma más inteligente:

  * Mismo nodo de origen que los positivos.
  * Evita repetir conexiones imposibles o triviales.
  * Produce *negativos duros* que ayudan al modelo a aprender mejor.

### 4. **Evaluación más clara**

* Se muestra:

  * Curva de pérdida por época.
  * Histograma de scores para positivos y negativos.
  * Métricas de `ROC AUC` y `Accuracy`.

### 5. **Función de predicción directa**

* `predict_link(origen_nombre, destino_nombre)` permite consultar la probabilidad de conexión para cualquier par.

---

## 🌐 Estructura

* `link_prediction.py`: Script principal.
* `models/gcn_link_model.pt`: Modelo GCN entrenado.
* `../../data/graph_link_prediction.pt`: Grafo PyTorch Geometric con embeddings.
* `../../data/label_encoder.pkl`: Codificador de nodos.

---

## 📊 Resultados esperados

* AUC > 0.89
* Accuracy mejorable, pero con solapamiento reducido de scores.
* Menor sobreajuste gracias a features y regularización.

---

## ✅ Próximos pasos recomendados

* Añadir más features: latitud/longitud, población, clúster regional.
* Comparar con otros modelos: GAT, GraphSAGE.
* Visualizar embeddings con PCA o UMAP.
* Hacer tabla de predicciones reales y su presencia en los datos.

---

