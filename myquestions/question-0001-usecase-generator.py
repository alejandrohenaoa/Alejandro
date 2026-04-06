import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def detectar_anomalias_industriales(df, contaminacion=0.05):
    """
    Detecta registros atípicos optimizando la precisión mediante escalado.
    """
    # 1. Selección de datos numéricos
    datos_numericos = df.select_dtypes(include=[np.number])
    
    if datos_numericos.empty:
        return np.array([])

    # 2. Imputación (Mediana para evitar sesgos de valores extremos)
    imputador = SimpleImputer(strategy='median')
    datos_limpios = imputador.fit_transform(datos_numericos)

    # 3. Escalado de datos (FUNDAMENTAL para Isolation Forest)
    # Esto asegura que la velocidad y el consumo se midan en la misma escala
    escalador = StandardScaler()
    datos_escalados = escalador.fit_transform(datos_limpios)

    # 4. Configuración del modelo
    modelo = IsolationForest(contamination=contaminacion, random_state=42)

    # 5. Ajuste y predicción
    return modelo.fit_predict(datos_escalados)

# --- Caso de Uso: Monitoreo de Flota ---

data = {
    'id_camion': range(1, 11),
    'velocidad_promedio_kmh': [80, 82, 79, 85, 20, 81, 83, 78, 90, 15],
    'consumo_combustible_gals': [10, 11, 10.5, 12, 45, 10.8, 11.2, 9.5, 13, 50]
}
df_logistica = pd.DataFrame(data)

# Llamada a la función original
resultados = detectar_anomalias_industriales(
    df_logistica[['velocidad_promedio_kmh', 'consumo_combustible_gals']], 
    contaminacion=0.2
)

df_logistica['estado'] = np.where(resultados == 1, 'Normal', 'Anomalía')

print("Análisis de la Flota:")
print(df_logistica)
