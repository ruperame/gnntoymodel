import joblib

le = joblib.load('../../data/label_encoder.pkl')
print("Total de nodos codificados:", len(le.classes_))
print("Primeros 20 nodos:", le.classes_[:20])
print("¿'SAO PAULO/SP' está en el grafo?", 'SAO PAULO/SP' in le.classes_)
print("¿'RIO DE JANEIRO/RJ' está en el grafo?", 'RIO DE JANEIRO/RJ' in le.classes_)
