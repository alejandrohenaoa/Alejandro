import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def use_case_generator(textos, etiquetas, nuevo_mensaje):
    # Limpieza de texto (minúsculas y tildes)
    def limpiar(t):
        t = t.lower()
        t = re.sub(r'[^\w\s]', '', t)
        return t.replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u')

    textos_limpios = [limpiar(msg) for msg in textos]
    
    # Vectorización y entrenamiento
    vectorizador = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizador.fit_transform(textos_limpios)
    
    modelo = MultinomialNB()
    modelo.fit(X, etiquetas)
    
    # Predicción del nuevo mensaje
    X_nuevo = vectorizador.transform([limpiar(nuevo_mensaje)])
    return modelo.predict(X_nuevo)

# Lógica de respuesta
if resultado_array[0] == "Seguridad":
    print("Acción: Bloqueo preventivo de cuenta activado.")
else:
    print("Acción: Remitir a preguntas frecuentes.")
