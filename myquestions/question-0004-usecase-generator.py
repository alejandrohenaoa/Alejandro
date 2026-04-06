import numpy as np
from sklearn.ensemble import RandomForestRegressor

def use_case_generator(X_train, y_train, X_test):
    # Configuración del modelo de regresión
    modelo = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    
    # Entrenamiento con datos históricos
    modelo.fit(X_train, y_train)
    
    # Retorno de predicciones como array
    return np.array(modelo.predict(X_test))
