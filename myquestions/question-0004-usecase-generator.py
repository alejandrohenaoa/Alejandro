import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def predecir_emisiones_co2(X_train, y_train, X_test):
    """
    Entrena un bosque aleatorio para estimar emisiones de carbono,
    incluyendo optimización de hiperparámetros básicos.
    """
    # 1. Configuración robusta del modelo
    # Aumentamos n_estimators y limitamos la profundidad (max_depth) 
    # para evitar que el modelo "memorice" en lugar de aprender.
    modelo = RandomForestRegressor(
        n_estimators=200, 
        max_depth=10,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1 # Usa todos los núcleos del procesador
    )

    # 2. Entrenamiento
    modelo.fit(X_train, y_train)

    # 3. Predicción
    predicciones = modelo.predict(X_test)

    # --- Mejora: Mostrar importancia de variables (Opcional pero recomendado) ---
    importancias = pd.Series(modelo.feature_importances_, index=X_train.columns)
    print("\n[INFO] Importancia de los factores en la emisión:")
    print(importancias.sort_values(ascending=False))

    # 4. Retorno como array de NumPy
    return np.array(predicciones)

# --- Ejemplo de aplicación: Inventario Ambiental ---

data_historica = {
    'consumo_kwh': [5000, 12000, 8000, 20000, 15000],
    'superficie_m2': [1000, 2500, 1800, 4000, 3200],
    'empleados': [50, 120, 80, 200, 150]
}
emisiones_reales = [150.5, 380.2, 240.0, 610.8, 450.3]

X_entrenamiento = pd.DataFrame(data_historica)
y_entrenamiento = np.array(emisiones_reales)

data_nuevas = {
    'consumo_kwh': [7000, 18000, 9500],
    'superficie_m2': [1500, 3500, 2000],
    'empleados': [70, 180, 90]
}
X_nuevas = pd.DataFrame(data_nuevas)

# Ejecución
emisiones_estimadas = predecir_emisiones_co2(X_entrenamiento, y_entrenamiento, X_nuevas)

print("\nEstimaciones de CO2 proyectadas:")
for i, est in enumerate(emisiones_estimadas):
    # Uso de f-strings para un reporte limpio
    print(f">> Planta Nueva {i+1}: {est:,.2f} Toneladas")
