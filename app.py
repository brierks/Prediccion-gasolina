
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Cargar modelo y encoder
model = joblib.load("modelo_gasolina.joblib")
encoder = joblib.load("encoder_gasolina.joblib")

# Interfaz
st.title("Predicción del Precio de la Gasolina Regular en México")

# Inputs
estado = st.selectbox("Selecciona un estado:", encoder.categories_[0])
anio = st.number_input("Selecciona el año:", min_value=2017, max_value=2025, value=2024, step=1)
mes = st.slider("Selecciona el mes:", 1, 12, 6)

# Preparar datos
X_estado = encoder.transform([[estado]]).toarray()
X_input = np.concatenate([X_estado, np.array([[anio, mes]])], axis=1)

# Predicción
prediccion = model.predict(X_input)[0]
st.subheader(f"Precio estimado: ${prediccion:.2f} MXN")
