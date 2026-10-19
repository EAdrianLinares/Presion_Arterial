import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Riesgo de presión arterial elevada", page_icon="🩺", layout="wide")


# ---------------------------------------------------------------------------
# Carga de archivos (se hace una sola vez y queda en caché)
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelos():
    modelo_lr = joblib.load("modelo_presion.pkl")
    modelo_arbol = joblib.load("modelo_arbol.pkl")
    with open("info_modelo.json", encoding="utf-8") as f:
        info = json.load(f)
    return modelo_lr, modelo_arbol, info


@st.cache_data
def cargar_datos():
    datos = pd.read_csv("datos_limpios.csv")
    resultados = pd.read_csv("resultados_prueba.csv")
    knn = pd.read_csv("comparacion_knn.csv")
    return datos, resultados, knn


modelo_lr, modelo_arbol, info = cargar_modelos()
datos, resultados, comparacion_knn = cargar_datos()
UMBRAL = info["umbral"]


# ---------------------------------------------------------------------------
# Página 1: Predicción
# ---------------------------------------------------------------------------
NIVELES_INGRESO = {
    "Bajo (no alcanza para lo básico)": 0.8,
    "Medio-bajo": 1.5,
    "Medio": 2.5,
    "Medio-alto": 3.5,
    "Alto": 5.0,
}


def pagina_prediccion():
    st.title("🩺 Estimador de riesgo de presión arterial elevada")
    st.caption("Proyecto académico de Machine Learning clásico · Especialización en IA (CUN) · "
               "Datos: NHANES 2015-2016 (CDC)")

    with st.sidebar:
        st.header("Datos de la persona")
        edad = st.slider("Edad", 18, 80, 45)
        sexo = st.radio("Sexo", ["Hombre", "Mujer"], horizontal=True)
        peso = st.number_input("Peso (kg)", 30.0, 250.0, 75.0, step=0.5)
        talla = st.number_input("Estatura (cm)", 120.0, 220.0, 168.0, step=0.5)
        cintura = st.number_input("Circunferencia de cintura (cm)", 50.0, 180.0, 95.0, step=0.5,
                                  help="Medida con cinta métrica a la altura del hueso de la cadera.")
        fuma = st.radio("¿Ha fumado al menos 100 cigarrillos en su vida?", ["No", "Sí"], horizontal=True)
        alcohol = st.radio("¿Ha tomado 12 o más tragos de alcohol en algún año?",
                           ["No", "Sí", "Prefiero no responder"], horizontal=True)
        if edad < 20:
            educacion = estado_civil = "No aplica (menor de 20)"
            st.info("Para menores de 20 años, educación y estado civil no aplican.")
        else:
            educacion = st.selectbox("Nivel educativo", info["opciones"]["educacion"], index=2)
            estado_civil = st.selectbox("Estado civil", info["opciones"]["estado_civil"])
        personas = st.slider("Personas que viven en el hogar", 1, 7, 3, help="7 significa 7 o más.")
        nivel_ingreso = st.select_slider("Nivel de ingreso del hogar", list(NIVELES_INGRESO), value="Medio")

    imc = peso / (talla / 100) ** 2
    entrada = pd.DataFrame([{
        "edad": edad, "imc": imc, "cintura_cm": cintura,
        "ratio_ingreso_pobreza": NIVELES_INGRESO[nivel_ingreso], "personas_hogar": personas,
        "sexo": sexo, "fuma": fuma,
        "alcohol": "No responde" if alcohol == "Prefiero no responder" else alcohol,
        "educacion": educacion, "estado_civil": estado_civil,
    }])

    col1, col2 = st.columns(2)
    col1.metric("IMC calculado", f"{imc:.1f}")
    clasificacion = ("Bajo peso" if imc < 18.5 else "Normal" if imc < 25
                     else "Sobrepeso" if imc < 30 else "Obesidad")
    col2.metric("Clasificación del IMC", clasificacion)

    if st.button("Calcular riesgo", type="primary", use_container_width=True):
        prob = modelo_lr.predict_proba(entrada)[0, 1]
        st.subheader(f"Probabilidad estimada de presión elevada: {prob:.0%}")
        st.progress(float(prob))
        if prob >= UMBRAL:
            st.error("**Riesgo ELEVADO.** Se recomienda medir la presión arterial "
                     "y consultar con un profesional de la salud.")
        else:
            st.success("**Riesgo BAJO** según el modelo. Aun así, se recomienda medir "
                       "la presión arterial de forma periódica.")
        st.caption(f"El modelo clasifica como riesgo elevado cuando la probabilidad es de {UMBRAL:.0%} o más. "
                   "Este umbral se eligió para detectar la mayor cantidad de casos posibles (tamizaje).")

    st.warning("⚠️ Esta herramienta es un ejercicio académico de tamizaje. "
               "**No es un diagnóstico médico** ni reemplaza la medición de la presión arterial.")


# ---------------------------------------------------------------------------
# Página 2: Los datos (EDA)  -> se construye en el siguiente paso
# ---------------------------------------------------------------------------
def pagina_datos():
    st.title("📊 Los datos")
    st.info("Sección en construcción.")
    st.write(f"Dataset limpio cargado: {datos.shape[0]:,} personas y {datos.shape[1]} columnas.")


# ---------------------------------------------------------------------------
# Página 3: Entrenamiento y evaluación  -> se construye después
# ---------------------------------------------------------------------------
def pagina_evaluacion():
    st.title("🧪 Entrenamiento y evaluación")
    st.info("Sección en construcción.")
    st.write(f"Resultados de prueba cargados: {len(resultados):,} personas.")


# ---------------------------------------------------------------------------
# Navegación entre páginas
# ---------------------------------------------------------------------------
navegacion = st.navigation([
    st.Page(pagina_prediccion, title="Predicción", icon="🩺", default=True),
    st.Page(pagina_datos, title="Los datos", icon="📊"),
    st.Page(pagina_evaluacion, title="Entrenamiento y evaluación", icon="🧪"),
])
navegacion.run()
