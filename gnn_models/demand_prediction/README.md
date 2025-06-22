# 📈 Modelo GNN de Predicción de Demanda (demand_prediction.py)

Este modelo implementa una **regresión de demanda** entre pares origen-destino con redes neuronales sobre grafos (**GNN**), utilizando datos de viajes de autobús en Brasil. El objetivo es predecir el número de viajes (`num_viajes`) entre dos puntos dados, teniendo en cuenta la línea, la hora y el día.

---

## 🎯 Objetivo

Predecir el **número de viajes (demanda)** entre un `ponto_origem_viagem` y un `ponto_destino_viagem`, según la línea de autobús (`nu_linha`), la **hora del día** y el **día de la semana**.

---

## 🧱 Pipeline de trabajo

1. **Carga y limpieza**  
   Se parte del dataset limpio `viajes_limpios.csv`, generado en el notebook `exploratory.ipynb`.

2. **Feature engineering**  
   - Se extrae `hora` y `dia_semana` a partir de `hora_viagem` y `data_viagem`.
   - Se agregan los viajes con `groupby` por (`origen`, `destino`, `línea`, `hora`, `día_semana`), calculando `num_viajes`.

3. **Eliminación de outliers**  
   - Se eliminan los pares con demanda superior al percentil 99 para reducir la varianza y mejorar la generalización.

4. **Codificación de nodos**  
   - Se aplica `LabelEncoder` para convertir los nombres de parada a IDs enteros.
   - Se guardan los codificadores para usar en visualizaciones posteriores.

5. **Normalización de features de arista**  
   - Las variables `nu_linha`, `hora` y `dia_semana` se escalan con `StandardScaler`.

6. **Transformación del target (demanda)**  
   - Se aplica `log1p(num_viajes)` para suavizar la distribución.
   - Durante la evaluación se invierte con `expm1()`.

7. **Separación en train y test**
   - Se divide el conjunto en 80% para entrenamiento y 20% para validación aleatoria.

8. **Definición del grafo PyTorch Geometric**
   - Nodos: paradas
   - Aristas: viajes únicos (origen-destino-línea-hora-día)
   - `edge_attr`: línea, hora, día
   - `edge_label`: demanda (número de viajes, log-transformado)

9. **Modelo GNN**  
   - GCN con 2 capas (`GCNConv`)
   - MLP final para regresión, usando los embeddings de los nodos origen y destino concatenados con los atributos de la arista.

10. **Entrenamiento**
    - 100 epochs
    - Optimización con `Adam` y pérdida `MSELoss`

11. **Evaluación y visualización**
    - Métricas: MSE y R²
    - Dispersión de predicciones vs valores reales
    - Se invierte la transformación logarítmica para interpretar los resultados en unidades reales

---

## 🧪 Cambios y mejoras incorporadas

| Mejora incorporada                  | Motivo                                                      |
|-------------------------------------|--------------------------------------------------------------|
| Eliminación del 1% superior de viajes | Reducir el impacto de outliers extremos                      |
| Transformación `log1p(num_viajes)` | Normalizar la escala y suavizar valores grandes              |
| Ingeniería de features: día, hora   | Incluir contexto temporal básico                             |
| Normalización con `StandardScaler` | Asegurar distribución equilibrada de los features numéricos |
| División en train/test              | Evaluación más fiable y realista                             |

---

## 📉 Resultados

### ❌ Primer intento (sin limpieza, sin log):
- **MSE**: `407547.41`
- **R²**: `-1.99`
- ➤ El modelo colapsó y las predicciones eran nulas o enormes (gráfica saturada)

### ❌ Segundo intento (log y normalización):
- **MSE**: `135487.11`
- **R²**: `0.00`
- ➤ Ligera mejora, pero aún predicciones agrupadas en pocos valores

### ✅ Tercer intento (con eliminación de outliers y test set separado):
- **MSE**: `4738.76`
- **R²**: `-0.67`
- ➤ Se ajustó mejor a los valores pequeños, pero no captura correctamente la variabilidad → el modelo **todavía no generaliza bien**.

---

## 📂 Archivos generados

| Archivo                                  | Descripción                               |
|------------------------------------------|-------------------------------------------|
| `graph_demand_prediction.pt`             | Grafo PyG con nodos, aristas y etiquetas  |
| `label_encoder_demand.pkl`               | Codificador de paradas                    |
| `models/gcn_demand_model.pt`             | Pesos del modelo entrenado                |

---

## 🚧 Próximos pasos sugeridos

- Añadir features de nodo (frecuencia, centralidad, etc.)
- Probar con GATConv (atención) o redes de mayor profundidad
- Usar escalado tipo `MinMaxScaler` en lugar de log
- Agrupar líneas o paradas similares para reducir dimensionalidad
- Balancear el conjunto para evitar dominancia de valores bajos

---

## ▶️ Ejecución

```bash
cd gnn_models/demand_prediction
python demand_prediction.py
