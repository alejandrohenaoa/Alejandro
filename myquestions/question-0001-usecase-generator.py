import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

def generar_caso_de_uso_detectar_anomalias_industriales(df, contaminacion=0.05):
    # Aseguramos que solo trabajamos con datos numéricos
    datos_numericos = df.select_dtypes(include=[np.number])

    # Manejo de valores nulos (Imputación por mediana)
    imputador = SimpleImputer(strategy='median')
    datos_limpios = imputador.fit_transform(datos_numericos)

    # Configuración y entrenamiento del modelo Isolation Forest
    modelo = IsolationForest(contamination=contaminacion, random_state=42)

    # El método fit_predict devuelve 1 para inliers y -1 para outliers
    predicciones = modelo.fit_predict(datos_limpios)

    return predicciones
