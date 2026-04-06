import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def use_case_generator(df, contaminacion=0.2):
    # Seleccionamos solo números para evitar errores de texto
    datos_numericos = df.select_dtypes(include=[np.number])
    
    # Escalado para mejorar la detección
    escalador = StandardScaler()
    datos_escalados = escalador.fit_transform(datos_numericos)
    
    # Modelo con el nombre de función requerido
    modelo = IsolationForest(contamination=contaminacion, random_state=42)
    return modelo.fit_predict(datos_escalados)
