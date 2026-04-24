"""
Interfaz Web — Laboratorio Minería de Datos
Modelos: Precio Dólar | Nivel Glucosa | Consumo Energía
Ejecutar: streamlit run app_streamlit.py
"""

import numpy as np
import streamlit as st
import joblib

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Predicciones Inteligentes — Minería de Datos",
    page_icon="🤖",
    layout="wide"
)

# CSS Personalizado para mejorar la estética y visibilidad
st.markdown("""
    <style>
    /* Forzar colores de texto para legibilidad */
    .stApp, .stMarkdown, p, h1, h2, h3, h4, span, label {
        color: #1e293b !important;
    }
    
    /* Fondo con degradado suave */
    .stApp {
        background: linear-gradient(135deg, #e2e8f0 0%, #f8fafc 100%);
    }
    
    /* Sidebar - Fondo blanco y texto oscuro */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #cbd5e1;
    }
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #334155 !important;
    }

    /* Estilo de Tarjetas/Contenedores */
    div[data-testid="stVerticalBlock"] > div > div > div[data-testid="stVerticalBlock"] {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        margin-bottom: 1rem;
        border: 1px solid #f1f5f9;
    }
    
    /* Título Principal */
    h1 {
        background: linear-gradient(to right, #1e40af, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        padding: 1rem 0;
    }
    
    /* Botones */
    .stButton>button {
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 2rem !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
    }
    
    /* Inputs y Sliders */
    .stNumberInput input, .stSelectbox div {
        background-color: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        color: #1e293b !important;
    }
    
    /* Métricas */
    div[data-testid="stMetricValue"] {
        color: #2563eb !important;
        font-weight: 800 !important;
    }

    /* Estilo para los Radio Buttons del Sidebar (Efecto Menú) */
    div[data-testid="stSidebar"] .stRadio > div {
        background-color: transparent !important;
        padding: 0 !important;
    }
    
    div[data-testid="stSidebar"] .stRadio label {
        background-color: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        padding: 12px 20px !important;
        border-radius: 12px !important;
        margin-bottom: 8px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: flex !important;
        align-items: center !important;
        cursor: pointer !important;
    }

    div[data-testid="stSidebar"] .stRadio label:hover {
        background-color: #f1f5f9 !important;
        border-color: #3b82f6 !important;
        transform: translateX(5px);
    }

    /* Ocultar el círculo del radio original para un look más limpio */
    div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    /* Estilo cuando está seleccionado */
    div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] input:checked + div {
        display: none !important;
    }
    
    div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-active="true"] {
        background: linear-gradient(90deg, #2563eb, #3b82f6) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
    }
    
    div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-active="true"] p {
        color: white !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BARRA LATERAL (SIDEBAR)
# ─────────────────────────────────────────────
with st.sidebar:
    # Logo o Imagen de perfil
    col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])
    with col_logo2:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", use_container_width=True)
    
    st.markdown("<h2 style='text-align: center; color: #1e293b; font-size: 1.5rem;'>Panel de Control</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.9rem;'>Configuración de Modelos</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🧠 Selección de Modelo")
    modelo_seleccionado = st.radio(
        "Selecciona el ejercicio a visualizar:",
        options=["💵 Precio del Dólar", "🩸 Nivel de Glucosa", "⚡ Consumo de Energía"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    with st.expander("ℹ️ Información del Sistema"):
        st.caption("""
        **Metodología:** CRISP-DM
        **Algoritmo:** Regresión Lineal Múltiple
        **Versión:** 2.0.1
        """)
    
    st.info("💡 Los cambios en la selección actualizan el panel principal automáticamente.")

# ─────────────────────────────────────────────
# CONTENIDO PRINCIPAL
# ─────────────────────────────────────────────
st.title("🤖 Laboratorio de Minería de Datos")
st.markdown("<p style='text-align: center;'>Desarrollado por Diego Ocampo y Dylan Rodíguez.</p>", unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# EJERCICIO 1 — PRECIO DEL DÓLAR
# ─────────────────────────────────────────────
if modelo_seleccionado == "💵 Precio del Dólar":
    st.header("💵 Predicción del Precio del Dólar")
    
    col_info, col_input = st.columns([1, 2], gap="large")
    
    with col_info:
        st.markdown("""
        ### Contexto
        Este modelo analiza variables macroeconómicas para predecir el comportamiento de la moneda.
        
        **Factores Clave:**
        - Tendencia histórica.
        - Estabilidad económica.
        - Tasas bancarias.
        """)
        st.warning("⚠️ Nota: Las predicciones son aproximaciones estadísticas.")

    with col_input:
        with st.container():
            st.subheader("Configuración de Variables")
            dia = st.number_input("📅 Día de la Serie", min_value=1, max_value=1000, value=250)
            inflacion = st.number_input("📈 Tasa de Inflación (decimal)", min_value=0.001, max_value=0.10, value=0.020, format="%.4f")
            tasa_interes = st.number_input("🏦 Tasa de Interés (%)", min_value=1.0, max_value=10.0, value=5.0, format="%.2f")

            if st.button("🚀 Calcular Valor Estimado", use_container_width=True):
                try:
                    import sklearn # Verificar si está instalado
                    modelo = joblib.load("modelo_dolar.pkl")
                    entrada = np.array([[dia, inflacion, tasa_interes]])
                    prediccion = modelo.predict(entrada)[0]
                    
                    st.divider()
                    st.metric(label="Precio Estimado", value=f"${prediccion:,.2f}")
                    st.success("✅ Predicción generada")
                except ImportError:
                    st.error("❌ Falta la librería `scikit-learn`. Por favor instálala con: `pip install scikit-learn`")
                except FileNotFoundError:
                    st.error("❌ Archivo `modelo_dolar.pkl` no encontrado.")

# ─────────────────────────────────────────────
# EJERCICIO 2 — NIVEL DE GLUCOSA
# ─────────────────────────────────────────────
elif modelo_seleccionado == "🩸 Nivel de Glucosa":
    st.header("🩸 Análisis de Glucosa")

    col_info, col_input = st.columns([1, 2], gap="large")

    with col_info:
        st.markdown("""
        ### Salud Preventiva
        Estimación de glucosa basada en indicadores biométricos.
        
        **Indicadores:**
        - Índice de Masa Corporal.
        - Actividad física semanal.
        - Edad del paciente.
        """)
        # Indicador visual de IMC
        imc_val = 25.0 # valor por defecto para el primer render
        if 'imc_input' in st.session_state: imc_val = st.session_state.imc_input
        
        if imc_val < 18.5: st.warning("⚠️ IMC: Bajo peso")
        elif imc_val < 25: st.success("✅ IMC: Normal")
        elif imc_val < 30: st.warning("⚠️ IMC: Sobrepeso")
        else: st.error("🔴 IMC: Obesidad")

    with col_input:
        with st.container():
            st.subheader("Datos del Paciente")
            edad = st.number_input("🎂 Edad (años)", min_value=18, max_value=100, value=45)
            imc = st.number_input("⚖️ IMC (kg/m²)", min_value=10.0, max_value=50.0, value=25.0, format="%.1f", key="imc_input")
            actividad = st.number_input("🏃 Actividad (h/semana)", min_value=0, max_value=40, value=3)

            if st.button("🚀 Estimar Glucosa", use_container_width=True):
                try:
                    import sklearn
                    modelo = joblib.load("modelo_glucosa.pkl")
                    entrada = np.array([[edad, imc, actividad]])
                    prediccion = modelo.predict(entrada)[0]
                    
                    st.divider()
                    st.metric(label="Glucosa Estimada", value=f"{prediccion:.2f} mg/dL")
                    if prediccion < 100: st.info("📋 Estado: **Normal**")
                    elif prediccion < 126: st.warning("📋 Estado: **Prediabetes**")
                    else: st.error("📋 Estado: **Posible Diabetes**")
                except ImportError:
                    st.error("❌ Falta la librería `scikit-learn`. Ejecuta: `pip install scikit-learn`")
                except FileNotFoundError:
                    st.error("❌ Archivo `modelo_glucosa.pkl` no encontrado.")

# ─────────────────────────────────────────────
# EJERCICIO 3 — CONSUMO DE ENERGÍA
# ─────────────────────────────────────────────
elif modelo_seleccionado == "⚡ Consumo de Energía":
    st.header("⚡ Gestión Energética")

    col_info, col_input = st.columns([1, 2], gap="large")

    with col_info:
        st.markdown("""
        ### Eficiencia Energética
        Pronóstico de demanda eléctrica según variables externas.
        
        **Factores:**
        - Clima y temperatura.
        - Ciclos horarios.
        - Día de la semana.
        """)

    with col_input:
        with st.container():
            st.subheader("Variables Ambientales")
            temperatura = st.number_input("🌡️ Temperatura (°C)", min_value=0.0, max_value=50.0, value=25.0)
            hora = st.slider("🕐 Hora del día", 1, 24, 12)
            dias_map = {"Lunes":1, "Martes":2, "Miércoles":3, "Jueves":4, "Viernes":5, "Sábado":6, "Domingo":7}
            dia_nombre = st.selectbox("📅 Día de la semana", options=list(dias_map.keys()))
            dia_semana = dias_map[dia_nombre]

            if st.button("🚀 Estimar Consumo", use_container_width=True):
                try:
                    import sklearn
                    modelo = joblib.load("modelo_energia.pkl")
                    hora_sin = np.sin(2 * np.pi * hora / 24)
                    hora_cos = np.cos(2 * np.pi * hora / 24)
                    entrada = np.array([[temperatura, hora_sin, hora_cos, dia_semana]])
                    prediccion = modelo.predict(entrada)[0]

                    st.divider()
                    st.metric(label="Consumo Estimado", value=f"{prediccion:.2f} kWh")
                    st.info(f"📊 Pronóstico para las {hora:02d}:00h del día {dia_nombre}.")
                except ImportError:
                    st.error("❌ Falta la librería `scikit-learn`. Ejecuta: `pip install scikit-learn`")
                except FileNotFoundError:
                    st.error("❌ Archivo `modelo_energia.pkl` no encontrado.")

# ─────────────────────────────────────────────
# PIE DE PÁGINA
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    """
    <div style='text-align:center; color:#7f8c8d; font-size:0.9em; padding: 20px;'>
        <hr style='border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));'>
        Laboratorio de Minería de Datos | Modelos Scikit-Learn | CRISP-DM 2026
    </div>
    """,
    unsafe_allow_html=True
)