import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

def generar_caso_de_uso_detectar_anomalias_industriales(contaminacion=0.05):
    
    df = pd.DataFrame({
        "temperatura": [50, 52, 49, 300, 51, 48, 47, 500],
        "presion": [30, 29, 31, 100, 30, 29, 28, 150],
        "vibracion": [0.2, 0.25, 0.22, 5.0, 0.21, 0.19, 0.2, 6.0]
    })

    # 🔢 Seleccionar columnas numéricas
    datos_numericos = df.select_dtypes(include=[np.number])

    # 🧹 Imputar valores nulos
    imputador = SimpleImputer(strategy='median')
    datos_limpios = pd.DataFrame(
        imputador.fit_transform(datos_numericos),
        columns=datos_numericos.columns
    )

    # 🤖 Modelo
    modelo = IsolationForest(contamination=contaminacion, random_state=42)

    # 🔍 Predicciones
    predicciones = modelo.fit_predict(datos_limpios)
    scores = modelo.decision_function(datos_limpios)

    # 📦 Resultado final
    df["anomalia"] = predicciones
    df["score_anomalia"] = scores

    return df
