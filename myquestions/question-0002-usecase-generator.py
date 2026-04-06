import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def comprimir_dimensiones_por_varianza(X, umbral_varianza=0.95):
    """
    Reduce la dimensionalidad de un dataset conservando un porcentaje
    específico de la varianza de los datos.
    """
    # 1. Filtro de seguridad: Solo columnas numéricas
    # El PCA y el escalado fallan si hay texto o IDs no numéricos.
    X_numeric = X.select_dtypes(include=[np.number])
    
    if X_numeric.empty:
        raise ValueError("El dataset no contiene columnas numéricas para procesar.")

    # 2. Escalamiento de datos
    # Crucial: PCA es sensible a las escalas (ej: cm vs km)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    # 3. Inicialización de PCA con el umbral de varianza
    pca = PCA(n_components=umbral_varianza, random_state=42)

    # 4. Ajuste y transformación
    X_reducido = pca.fit_transform(X_scaled)

    # 5. Conversión a DataFrame para mantener la estructura de Pandas
    # Creamos nombres de columnas tipo 'PC1', 'PC2', etc.
    n_componentes = X_reducido.shape[1]
    columnas_pca = [f'PC{i+1}' for i in range(n_componentes)]
    
    df_reducido = pd.DataFrame(X_reducido, columns=columnas_pca, index=X.index)

    return df_reducido

# --- Ejemplo de aplicación: Dataset de Sensores Biométricos ---

np.random.seed(42)
base_data = np.random.rand(100, 5) 
redundant_data = np.dot(base_data, np.random.rand(5, 45)) 
full_data = np.hstack([base_data, redundant_data])

# Añadimos una columna de texto (como un ID) para probar la robustez
df_biometrico = pd.DataFrame(full_data, columns=[f'Sensor_{i}' for i in range(50)])
df_biometrico['ID_Paciente'] = [f'P-{i}' for i in range(100)] 

print(f"Dimensiones originales: {df_biometrico.shape}")

# Ejecutamos la función corregida
df_final = comprimir_dimensiones_por_varianza(df_biometrico, umbral_varianza=0.90)

print(f"Dimensiones después de la compresión: {df_final.shape}")
print(f"Tipo de objeto devuelto: {type(df_final)}")
print("\nPrimeras 2 filas del DataFrame reducido:")
print(df_final.head(2))
