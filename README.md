# Modelo 4: Mejoras

Franjas horarias continuas relacionadas por distancia con sin/cos

Modelo multifranja: Antes entrenábamos un modelo por franja y línea. Ahora entrenamos un sólo modelo GCN que ve muchos grafos distintos (uno por franja y línea). La idea es que aprenda a generalizar. Escalable sin aumentar el entrenamiento, que sigue siendo robusto.

El modelo se entrena con 8 grafos (4 franjas x 2 líneas)