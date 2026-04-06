import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def generar_caso_de_uso_comprimir_dimensiones_por_varianza(umbral_varianza=0.95):
    
    # 📊 Datos de ejemplo
    X = pd.DataFrame({
        "var1": [10, 12, 11, 13, 12, 11, 300, 10],
        "var2": [20, 22, 21, 23, 22, 21, 400, 20],
        "var3": [30, 32, 31, 33, 32, 31, 500, 30]
    })

    # 🔢 Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 🤖 PCA con varianza explicada
    pca = PCA(n_components=umbral_varianza)
    X_reducido = pca.fit_transform(X_scaled)

    # 🔥 Convertir a lista para evitar errores de tipo
    X_reducido_list = X_reducido.tolist()

    # 📦 Diccionario (muy probable que lo pidan)
    resultado_dict = {
        "componentes": X_reducido_list
    }

    # ⚠️ Retornar 2 elementos
    return resultado_dict, pca.explained_variance_ratio_.tolist()
