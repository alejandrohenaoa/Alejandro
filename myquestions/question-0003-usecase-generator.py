import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def clasificar_tickets_soporte(textos, etiquetas, nuevo_mensaje):
    """
    Entrena un modelo de clasificación de texto con limpieza de datos
    y predice la categoría de un nuevo mensaje.
    """
    # 1. Preprocesamiento de texto (Limpieza)
    # Es vital para que 'Medellín' y 'medellin' se cuenten como la misma palabra.
    def limpiar_texto(t):
        t = t.lower() # Todo a minúsculas
        t = re.sub(r'[^\w\s]', '', t) # Quitar puntuación
        # Quitar tildes de forma manual para evitar dependencias externas
        t = t.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
        return t

    textos_limpios = [limpiar_texto(msg) for msg in textos]
    nuevo_mensaje_limpio = limpiar_texto(nuevo_mensaje)

    # 2. Vectorización TF-IDF
    # Agregamos ngram_range=(1,2) para captar frases como "informacion personal"
    vectorizador = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizador.fit_transform(textos_limpios)

    # 3. Entrenamiento del modelo Naive Bayes
    # MultinomialNB es excelente para conteo de palabras
    modelo = MultinomialNB()
    modelo.fit(X, etiquetas)

    # 4. Transformación y Predicción
    X_nuevo = vectorizador.transform([nuevo_mensaje_limpio])
    prediccion = modelo.predict(X_nuevo)

    return prediccion

# --- Ejemplo de aplicación: Clasificación de Intenciones ---

mensajes_entrenamiento = [
    "Mi cuenta fue hackeada y perdí mi acceso",
    "¿A qué hora cierran la tienda el sábado?",
    "Alguien entró en mi perfil sin permiso",
    "¿Cuál es el precio del servicio premium?",
    "Urgente: mi información personal está expuesta",
    "¿Tienen sucursales en Medellín?"
]

categorias = [
    "Seguridad", "Información", "Seguridad",
    "Información", "Seguridad", "Información"
]

# Mensaje de prueba con tildes y mayúsculas
nuevo_ticket = "¡URGENTE! Mi CONTRASEÑA fue robada en MEDELLÍN"

# Ejecución de la función
resultado_array = clasificar_tickets_soporte(mensajes_entrenamiento, categorias, nuevo_ticket)

print(f"Nuevo mensaje: '{nuevo_ticket}'")
print(f"Predicción del sistema: {resultado_array[0]}")

# Lógica de respuesta
if resultado_array[0] == "Seguridad":
    print("Acción: Bloqueo preventivo de cuenta activado.")
else:
    print("Acción: Remitir a preguntas frecuentes.")
