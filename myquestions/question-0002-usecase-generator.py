import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def generar_caso_de_uso_comprimir_dimensiones_por_varianza(X, umbral_varianza=0.95):
    # Escalar los datos antes de aplicar PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Inicializamos PCA con el umbral de varianza deseado
    pca = PCA(n_components=umbral_varianza)

    # Ajustamos y transformamos los datos
    X_reducido = pca.fit_transform(X_scaled)

    return X_reducido
