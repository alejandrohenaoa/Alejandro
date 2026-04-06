import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def generar_caso_de_uso_predecir_emisiones_co2(X_train, y_train, X_test):
    modelo = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    modelo.fit(X_train, y_train)
    return np.array(modelo.predict(X_test))
