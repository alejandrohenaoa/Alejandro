import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

def generar_caso_de_uso_detectar_anomalias_industriales(df, contaminacion=0.05):
    # 1. Filtro de datos numéricos
    datos_numericos = df.select_dtypes(include=[np.number])

    # 2. Imputación de nulos
    imputador = SimpleImputer(strategy='median')
    datos_limpios = imputador.fit_transform(datos_numericos)

    # 3. Modelo
    modelo = IsolationForest(contamination=contaminacion, random_state=42)

    # 4. Predicción
    return modelo.fit_predict(datos_limpios)
