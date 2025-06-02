# Modelo 3

En este tercer modelo se busca añadir una componente temporal al proyecto, añadiendo franjas horarios al recorrido.



## Enfoque práctico
Vamos a crear una copia del grafo por franja horaria.
Añadiremos un nuevo feature por nodo que codifica la franja horaria.
Entrenamos un modelo por franja o preparamos los datos para entrenamiento conjunto más adelante.

## Posibles features extra
Introducir predicción multi-nodo (predecir demanda en todas las paradas).
Añadir features como: día de semana, clima, eventos.