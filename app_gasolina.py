import numpy as np
import streamlit as st
import pandas as pd
import joblib

# --- Configuración de la página ---
st.set_page_config(
    page_title="Predicción Gasolina México",
    page_icon="⛽",
    layout="centered"
)

# --- Estilos CSS personalizados ---
st.markdown("""
    <style>
        .main {
            background-color: #0f1116;
            color: #ffffff;
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 2.2em;
            font-weight: bold;
            color: #00c4ff;
            margin-bottom: 0.5em;
        }
        .subtitle {
            text-align: center;
            font-size: 1.1em;
            color: #d1d1d1;
            margin-bottom: 2em;
        }
        .stImage > img {
            border-radius: 12px;
        }
        .result-box {
            background: #1c1f26;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            margin-top: 20px;
            border: 1px solid #00c4ff33;
        }
        .result-price {
            font-size: 1.8em;
            font-weight: bold;
            color: #00ff9f;
        }
    </style>
""", unsafe_allow_html=True)

# --- Título principal ---
st.markdown('<div class="title"> Predicción del Precio de la Gasolina en México</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Modelo entrenado con datos de la CRE</div>', unsafe_allow_html=True)

# --- Imagen central ---
st.image("Gasolina.jpg", caption="Predicción de precios con base en datos de la CRE", use_column_width=True)

st.markdown("---")

# --- Cargar modelo y encoder ---
modelo = joblib.load("modelo_gasolina.joblib")
encoder = joblib.load("encoder_gasolina.joblib")

# --- Cargar dataset tidy para lista de estados ---
df = pd.read_csv("precios_gasolina_tidy.csv")
estados = sorted(df['estado'].unique())

# --- Layout con columnas ---
st.header("Selecciona los parámetros")
col1, col2, col3 = st.columns(3)

with col1:
    estado = st.selectbox('Estado:', estados)

with col2:
    año = st.number_input('Año:', min_value=2017, max_value=2030, value=2023, step=1)

with col3:
    mes = st.number_input('Mes:', min_value=1, max_value=12, value=1, step=1)

# --- Transformar entrada ---
estado_encoded = encoder.transform([[estado]]).toarray()
entrada = np.concatenate([estado_encoded, [[año, mes]]], axis=1)

# --- Predicción ---
prediccion = modelo.predict(entrada)

# --- Resultado bonito ---
st.markdown('<div class="result-box">', unsafe_allow_html=True)
st.subheader("Precio estimado de la gasolina regular")
st.markdown(f'<div class="result-price">${prediccion[0]:.2f} MXN por litro</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
