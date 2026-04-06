import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def generar_caso_de_uso_clasificar_tickets_soporte(textos, etiquetas, nuevo_mensaje):
    # 1. Convertimos la lista de textos a un DataFrame
    df = pd.DataFrame({'mensaje': textos})

    # 2. Vectorización TF-IDF
    vectorizador = TfidfVectorizer(stop_words=None)
    X = vectorizador.fit_transform(df['mensaje'])

    # 3. Entrenamiento del modelo Naive Bayes
    modelo = MultinomialNB()
    modelo.fit(X, etiquetas)

    # 4. Transformar el nuevo mensaje y predecir
    X_nuevo = vectorizador.transform([nuevo_mensaje])
    prediccion = modelo.predict(X_nuevo)

    return prediccion
