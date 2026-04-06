import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def use_case_generator(X, umbral_varianza=0.95):
    # Filtrado de datos numéricos
    X_numeric = X.select_dtypes(include=[np.number])
    
    # Escalado esencial para PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    # Reducción de dimensiones
    pca = PCA(n_components=umbral_varianza, random_state=42)
    X_reducido = pca.fit_transform(X_scaled)
    
    return pd.DataFrame(X_reducido, index=X.index)
