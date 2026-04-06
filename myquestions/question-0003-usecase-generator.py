import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def generar_caso_de_uso_clasificar_tickets_soporte():
    
    # 📩 Datos de ejemplo (tickets de soporte)
    textos = [
        "No puedo iniciar sesión",
        "Error en el pago con tarjeta",
        "La aplicación se cierra sola",
        "Quiero cambiar mi contraseña",
        "No funciona el botón de compra",
        "Problema con la factura"
    ]

    etiquetas = [
        "login",
        "pago",
        "bug",
        "cuenta",
        "bug",
        "facturacion"
    ]

    nuevo_mensaje = "No puedo pagar mi suscripción"

    # 🔤 Vectorización TF-IDF
    vectorizador = TfidfVectorizer()
    X = vectorizador.fit_transform(textos)

    # 🤖 Modelo Naive Bayes
    modelo = MultinomialNB()
    modelo.fit(X, etiquetas)

    # 🔍 Transformar nuevo mensaje
    X_nuevo = vectorizador.transform([nuevo_mensaje])

    # 📊 Predicción
    prediccion = modelo.predict(X_nuevo)[0]

    # 📈 Probabilidades (extra pro 🔥)
    probabilidades = modelo.predict_proba(X_nuevo)[0]

    # 📦 Diccionario (OBLIGATORIO como primer elemento)
    resultado_dict = {
        "mensaje": nuevo_mensaje,
        "categoria_predicha": prediccion
    }

    # ⚠️ Retornar 2 elementos
    return resultado_dict, probabilidades.tolist()
