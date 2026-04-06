# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

def generar_caso_de_uso_detectar_anomalias_industriales(df, contaminacion=0.05):
    """
    Detecta registros atípicos en un DataFrame utilizando Isolation Forest.
    """
    # 1. Aseguramos que solo trabajamos con datos numéricos
    datos_numericos = df.select_dtypes(include=[np.number])

    # 2. Manejo de valores nulos (Imputación por mediana)
    imputador = SimpleImputer(strategy='median')
    datos_limpios = imputador.fit_transform(datos_numericos)

    # 3. Configuración y entrenamiento del modelo
    # random_state asegura que el resultado sea reproducible
    modelo = IsolationForest(contamination=contaminacion, random_state=42)

    # 4. Ajuste y predicción
    predicciones = modelo.fit_predict(datos_limpios)

    return predicciones
