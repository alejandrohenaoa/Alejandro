import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def generar_caso_de_uso_predecir_emisiones_co2(X_train, y_train, X_test):
    # Instanciar el modelo de regresión
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)

    # Entrenar el modelo con los datos históricos
    modelo.fit(X_train, y_train)

    # Realizar las predicciones sobre el conjunto nuevo
    predicciones = modelo.predict(X_test)

    return np.array(predicciones)
