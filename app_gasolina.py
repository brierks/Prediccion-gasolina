import numpy as np
import streamlit as st
import pandas as pd
import joblib

st.write('''# Predicción del Precio de la Gasolina en México''')
st.image("Gasolina.jpg", caption="Predicción de precios con base en datos de la CRE", width=150)

st.header('Selecciona los parámetros')

# --- Cargar modelo y encoder ---
modelo = joblib.load("modelo_gasolina.joblib")
encoder = joblib.load("encoder_gasolina.joblib")

# --- Cargar dataset tidy para lista de estados ---
df = pd.read_csv("precios_gasolina_tidy.csv")
estados = df['estado'].unique()

# --- Entradas del usuario ---
estado = st.selectbox('Estado:', estados)
año = st.number_input('Año:', min_value=2017, max_value=2030, value=2023, step=1)
mes = st.number_input('Mes:', min_value=1, max_value=12, value=1, step=1)

# --- Transformar entrada ---
estado_encoded = encoder.transform([[estado]]).toarray()
entrada = np.concatenate([estado_encoded, [[año, mes]]], axis=1)

# --- Predicción ---
prediccion = modelo.predict(entrada)

st.subheader('Precio estimado de la gasolina regular')
st.write(f" **${prediccion[0]:.2f} MXN por litro**")

