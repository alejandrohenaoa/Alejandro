import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def generar_caso_de_uso_predecir_emisiones_co2():
    
    # 📊 Datos de ejemplo (simulando variables industriales)
    X_train = pd.DataFrame({
        "produccion": [100, 150, 200, 250, 300, 350],
        "energia": [50, 60, 70, 80, 90, 100],
        "transporte": [20, 25, 30, 35, 40, 45]
    })

    y_train = [200, 240, 300, 360, 420, 480]  # emisiones CO2

    X_test = pd.DataFrame({
        "produccion": [275, 320],
        "energia": [85, 95],
        "transporte": [38, 42]
    })

    # 🤖 Modelo
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)

    # Entrenamiento
    modelo.fit(X_train, y_train)

    # 🔍 Predicciones
    predicciones = modelo.predict(X_test)

    # 📦 Diccionario (OBLIGATORIO)
    resultado_dict = {
        "predicciones_co2": predicciones.tolist()
    }

    # ⚠️ Retornar 2 elementos
    return resultado_dict, predicciones.tolist()
