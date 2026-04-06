import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def generar_caso_de_uso_clasificar_tickets_soporte(textos, etiquetas, nuevo_mensaje):
    # Convertimos la lista de textos a un DataFrame para manejo consistente
    df = pd.DataFrame({'mensaje': textos})

    # Vectorización TF-IDF
    vectorizador = TfidfVectorizer(stop_words=None)
    X = vectorizador.fit_transform(df['mensaje'])

    # Entrenamiento del modelo Naive Bayes
    modelo = MultinomialNB()
    modelo.fit(X, etiquetas)

    # Transformar el nuevo mensaje usando el mismo vectorizador
    X_nuevo = vectorizador.transform([nuevo_mensaje])

    # Realizar la predicción
    prediccion = modelo.predict(X_nuevo)

    return prediccion
