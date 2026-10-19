import json
import joblib
import pandas as pd
import plotly.express as px
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
# Página 2: Los datos (EDA)
# ---------------------------------------------------------------------------
NUMERICAS = ["edad", "imc", "cintura_cm", "ratio_ingreso_pobreza", "personas_hogar"]
CATEGORICAS = ["sexo", "fuma", "alcohol", "educacion", "estado_civil"]

DICCIONARIO = pd.DataFrame([
    ("edad", "RIDAGEYR", "Edad en años (80 = 80 o más)", "Numérica", "Entrevista"),
    ("sexo", "RIAGENDR", "Hombre / Mujer", "Categórica", "Entrevista"),
    ("imc", "BMXBMI", "Índice de masa corporal: peso / estatura² (kg/m²)", "Numérica", "Examen físico"),
    ("cintura_cm", "BMXWAIST", "Circunferencia de cintura (cm)", "Numérica", "Examen físico"),
    ("fuma", "SMQ020", "¿Ha fumado al menos 100 cigarrillos en su vida?", "Categórica", "Entrevista"),
    ("alcohol", "ALQ101", "¿Ha tomado 12 o más tragos en algún año?", "Categórica", "Entrevista"),
    ("educacion", "DMDEDUC2", "Máximo nivel educativo (mayores de 20 años)", "Categórica", "Entrevista"),
    ("estado_civil", "DMDMARTL", "Estado civil (mayores de 20 años)", "Categórica", "Entrevista"),
    ("personas_hogar", "DMDHHSIZ", "Personas que viven en el hogar (7 = 7 o más)", "Numérica", "Entrevista"),
    ("ratio_ingreso_pobreza", "INDFMPIR", "Ingreso familiar / línea de pobreza (5 = 5 o más)", "Numérica", "Calculada"),
    ("presion_elevada", "BPXSY1-2, BPXDI1-2", "Objetivo: 1 si la presión promedio es ≥130/80 mm Hg", "Binaria",
     "Examen físico"),
], columns=["Variable", "Código NHANES", "Descripción", "Tipo", "Origen"])

COLORES = {"Normal": "#4C9F70", "Elevada": "#D1495B"}


def pagina_datos():
    st.title("📊 Los datos")
    st.write("El modelo se entrenó con un subconjunto de la **Encuesta Nacional de Salud y Nutrición "
             "de EE. UU. (NHANES 2015-2016)**, realizada por el CDC. Combina una **entrevista** "
             "(datos que la persona reporta) con un **examen físico** (datos medidos por personal entrenado).")

    df = datos.copy()
    df["Presión"] = df["presion_elevada"].map({0: "Normal", 1: "Elevada"})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Personas (tras la limpieza)", f"{len(df):,}")
    c2.metric("Variables predictoras", len(NUMERICAS) + len(CATEGORICAS))
    c3.metric("Con presión elevada", f"{df['presion_elevada'].mean():.0%}")
    c4.metric("Edad mediana", f"{df['edad'].median():.0f} años")

    with st.expander("📖 Diccionario de las variables usadas"):
        st.dataframe(DICCIONARIO, hide_index=True, use_container_width=True)
        st.caption("Se descartaron 18 de las 28 columnas originales: identificadores y variables de diseño "
                   "de la encuesta, las presiones (definen el objetivo: fuga de datos), variables redundantes "
                   "con el IMC, columnas con demasiados nulos y medidas poco prácticas para la app.")

    # --- Variable objetivo -------------------------------------------------
    st.header("1. Variable objetivo")
    col_a, col_b = st.columns([1, 2])
    conteo = df["Presión"].value_counts().reset_index()
    conteo.columns = ["Presión", "Personas"]
    fig = px.pie(conteo, names="Presión", values="Personas", hole=0.45,
                 color="Presión", color_discrete_map=COLORES)
    col_a.plotly_chart(fig, use_container_width=True)
    col_b.markdown(
        "Se definió **presión elevada** con el criterio clínico ACC/AHA 2017: sistólica promedio "
        "**≥ 130 mm Hg** o diastólica promedio **≥ 80 mm Hg**, usando el promedio de las dos lecturas "
        "del examen físico.\n\n"
        f"Las clases están razonablemente balanceadas (**{df['presion_elevada'].mean():.0%}** elevada), "
        "pero aun así se usó `class_weight='balanced'` y se priorizó el **recall**, porque en salud "
        "el error más grave es no detectar a una persona con presión alta."
    )

    # --- Exploración interactiva --------------------------------------------
    st.header("2. Explora una variable")
    variable = st.selectbox("Elige una variable", NUMERICAS + CATEGORICAS,
                            format_func=lambda v: v.replace("_", " ").capitalize())

    if variable in NUMERICAS:
        col_a, col_b = st.columns(2)
        fig = px.histogram(df, x=variable, color="Presión", barmode="overlay", nbins=40, opacity=0.65,
                           color_discrete_map=COLORES, title=f"Distribución de {variable}")
        col_a.plotly_chart(fig, use_container_width=True)
        fig = px.box(df, x="Presión", y=variable, color="Presión", color_discrete_map=COLORES,
                     title=f"{variable} según la presión")
        col_b.plotly_chart(fig, use_container_width=True)
        medianas = df.groupby("Presión")[variable].median()
        st.caption(f"Mediana con presión normal: **{medianas.get('Normal', float('nan')):.1f}** · "
                   f"con presión elevada: **{medianas.get('Elevada', float('nan')):.1f}**")
    else:
        tasa = (df.dropna(subset=[variable])
                  .groupby(variable)["presion_elevada"].agg(["mean", "count"]).reset_index())
        tasa.columns = [variable, "tasa", "personas"]
        tasa = tasa.sort_values("tasa")
        fig = px.bar(tasa, x="tasa", y=variable, orientation="h", text=tasa["tasa"].map("{:.0%}".format),
                     hover_data=["personas"], color_discrete_sequence=["#D1495B"],
                     title=f"% con presión elevada según {variable}")
        fig.update_xaxes(tickformat=".0%", title="% con presión elevada")
        st.plotly_chart(fig, use_container_width=True)

    # --- Correlación ----------------------------------------------------------
    st.header("3. Correlación entre variables numéricas")
    corr = df[NUMERICAS + ["presion_elevada"]].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("La **edad** es la variable con mayor correlación con la presión elevada, seguida de "
               "la cintura y el IMC. La cintura y el IMC están muy correlacionadas entre sí.")

    # --- Valores nulos ---------------------------------------------------------
    st.header("4. Valores nulos y cómo se trataron")
    nulos = df[NUMERICAS + CATEGORICAS].isna().sum()
    nulos = nulos[nulos > 0].sort_values().reset_index()
    nulos.columns = ["Variable", "Nulos"]
    fig = px.bar(nulos, x="Nulos", y="Variable", orientation="h", text="Nulos",
                 color_discrete_sequence=["#EDAE49"])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "- **Numéricas** (cintura, IMC, ingreso): se imputaron con **KNN** (5 vecinos más parecidos), "
        "después de dividir en entrenamiento y prueba.\n"
        "- **Categóricas** (alcohol, tabaco): los vacíos pasaron a una categoría propia, *No responde*.\n"
        "- **Educación y estado civil**: a los menores de 20 años no se les hacía la pregunta, así que "
        "se les asignó *No aplica (menor de 20)* en lugar de un valor inventado."
    )

    # --- KNN vs mediana -------------------------------------------------------
    st.header("5. ¿Por qué KNN y no la mediana?")
    st.write("A las personas sin cintura medida les falta el dato con más frecuencia si son mayores y tienen "
             "más peso. Rellenarlas con la mediana subestimaría su riesgo; KNN asigna una cintura coherente "
             "con su IMC.")
    knn = comparacion_knn.dropna(subset=["imc"])
    col_a, col_b = st.columns([2, 1])
    fig = px.scatter(knn, x="imc", y="cintura_knn", opacity=0.7,
                     labels={"imc": "IMC de la persona", "cintura_knn": "Cintura imputada (cm)"},
                     color_discrete_sequence=["#30638E"])
    fig.add_hline(y=knn["cintura_mediana"].iloc[0], line_dash="dash", line_color="gray",
                  annotation_text="Imputación con la mediana")
    col_a.plotly_chart(fig, use_container_width=True)
    col_b.metric("Cintura promedio imputada con KNN", f"{comparacion_knn['cintura_knn'].mean():.1f} cm")
    col_b.metric("Mediana (método simple)", f"{comparacion_knn['cintura_mediana'].iloc[0]:.1f} cm")


# ---------------------------------------------------------------------------
# Página 3: Entrenamiento y evaluación
# ---------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

MODELOS = {"Regresión logística": "prob_logistica", "Árbol de decisión": "prob_arbol"}


def calcular_metricas(y_real, prob, umbral):
    pred = (prob >= umbral).astype(int)
    vn, fp, fn, vp = confusion_matrix(y_real, pred, labels=[0, 1]).ravel()
    precision = vp / (vp + fp) if (vp + fp) else 0.0
    recall = vp / (vp + fn) if (vp + fn) else 0.0
    return {
        "Accuracy": (vp + vn) / len(y_real),
        "Precisión": precision,
        "Recall": recall,
        "Especificidad": vn / (vn + fp) if (vn + fp) else 0.0,
        "F1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
        "VN": vn, "FP": fp, "FN": fn, "VP": vp,
    }


def grafico_matriz(m, titulo):
    matriz = [[m["VN"], m["FP"]], [m["FN"], m["VP"]]]
    fig = px.imshow(matriz, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                    x=["Predijo normal", "Predijo elevada"], y=["Realmente normal", "Realmente elevada"],
                    title=titulo)
    fig.update_layout(coloraxis_showscale=False)
    return fig


def pagina_evaluacion():
    st.title("🧪 Entrenamiento y evaluación")
    y = resultados["real"].values

    # --- Cómo se entrenó ----------------------------------------------------
    st.header("1. Cómo se entrenó")
    st.markdown(
        "- **División:** 80 % entrenamiento (4.440 personas) y 20 % prueba (1.110), estratificada.\n"
        "- **Preprocesamiento en un Pipeline:** escalado → imputación KNN para numéricas; "
        "categoría *No responde* + One-Hot para categóricas. Todo se aprende solo con entrenamiento.\n"
        "- **Modelos comparados:** regresión logística y árbol de decisión, ambos con "
        "`class_weight='balanced'`.\n"
        "- **Ajuste de hiperparámetros:** GridSearchCV con validación cruzada de 5 particiones, "
        "optimizando F1.\n"
        "- **Sobreajuste:** no se observó. La accuracy de entrenamiento y prueba difiere en ~1 punto "
        "(logística 0,673 vs 0,676; árbol 0,670 vs 0,660)."
    )

    # --- Comparación con umbral por defecto ---------------------------------
    st.header("2. Comparación de modelos (umbral 0,50)")
    filas = []
    for nombre, col in MODELOS.items():
        m = calcular_metricas(y, resultados[col].values, 0.5)
        filas.append({"Modelo": nombre, **{k: round(v, 3) for k, v in m.items()
                                          if k not in ("VN", "FP", "FN", "VP")},
                      "ROC-AUC": round(roc_auc_score(y, resultados[col]), 3)})
    st.dataframe(pd.DataFrame(filas).set_index("Modelo"), use_container_width=True)

    col_a, col_b = st.columns(2)
    for col_st, (nombre, col) in zip([col_a, col_b], MODELOS.items()):
        m = calcular_metricas(y, resultados[col].values, 0.5)
        col_st.plotly_chart(grafico_matriz(m, nombre), use_container_width=True)
    st.caption("Los **falsos negativos** (abajo a la izquierda) son el error más grave: personas con presión "
               "elevada a las que el modelo diría que están bien.")

    # --- Curva ROC -------------------------------------------------------------
    st.header("3. Curva ROC")
    curvas = []
    for nombre, col in MODELOS.items():
        fpr, tpr, _ = roc_curve(y, resultados[col])
        auc = roc_auc_score(y, resultados[col])
        curvas.append(pd.DataFrame({"Tasa de falsos positivos": fpr, "Tasa de verdaderos positivos": tpr,
                                    "Modelo": f"{nombre} (AUC {auc:.3f})"}))
    fig = px.line(pd.concat(curvas), x="Tasa de falsos positivos", y="Tasa de verdaderos positivos",
                  color="Modelo")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Mientras más se acerca la curva a la esquina superior izquierda, mejor ordena el modelo "
               "a las personas de menor a mayor riesgo. La diagonal gris equivale al azar.")

    # --- Umbral interactivo -----------------------------------------------------
    st.header("4. Ajuste del umbral de decisión")
    st.write("El modelo da una **probabilidad**; el umbral decide a partir de qué valor se clasifica como "
             "presión elevada. Mueve el umbral y observa el efecto en las métricas.")
    c1, c2 = st.columns([1, 2])
    modelo_sel = c1.radio("Modelo", list(MODELOS), key="modelo_umbral")
    umbral = c2.slider("Umbral", 0.20, 0.70, float(UMBRAL), 0.05)
    prob = resultados[MODELOS[modelo_sel]].values
    m = calcular_metricas(y, prob, umbral)
    base = calcular_metricas(y, prob, 0.5)

    cols = st.columns(5)
    for col_st, k in zip(cols, ["Accuracy", "Precisión", "Recall", "Especificidad", "F1"]):
        col_st.metric(k, f"{m[k]:.3f}", delta=f"{m[k] - base[k]:+.3f} vs 0,50")

    col_a, col_b = st.columns(2)
    col_a.plotly_chart(grafico_matriz(m, f"Matriz de confusión (umbral {umbral:.2f})"),
                       use_container_width=True)
    umbrales = [round(u, 2) for u in pd.Series(range(20, 71, 2)) / 100]
    curva = pd.DataFrame([{"Umbral": u, **{k: v for k, v in calcular_metricas(y, prob, u).items()
                                           if k in ("Recall", "Precisión", "Especificidad", "F1")}}
                          for u in umbrales]).melt(id_vars="Umbral", var_name="Métrica", value_name="Valor")
    fig = px.line(curva, x="Umbral", y="Valor", color="Métrica", title="Métricas según el umbral")
    fig.add_vline(x=umbral, line_dash="dash", line_color="gray")
    col_b.plotly_chart(fig, use_container_width=True)

    st.info(f"Con este umbral, el modelo deja sin detectar a **{m['FN']}** de las "
            f"{m['FN'] + m['VP']} personas con presión elevada (con 0,50 eran {base['FN']}) "
            f"y genera **{m['FP']}** falsas alarmas.")
    st.success(f"**Decisión final:** regresión logística con umbral **{UMBRAL}**. Logra el mejor F1 y un "
               "recall cercano a 0,89: prioriza detectar casos, aceptando más falsas alarmas, porque una "
               "medición extra de presión es barata y una hipertensión no detectada no lo es.")

    # --- Interpretación ----------------------------------------------------------
    st.header("5. ¿Qué aprendieron los modelos?")
    col_a, col_b = st.columns(2)
    nombres = modelo_lr.named_steps["prep"].get_feature_names_out()
    coef = pd.DataFrame({"Variable": [n.split("__", 1)[1] for n in nombres],
                         "Coeficiente": modelo_lr.named_steps["modelo"].coef_[0]})
    coef = coef.reindex(coef["Coeficiente"].abs().sort_values().index).tail(12)
    coef["Efecto"] = coef["Coeficiente"].map(lambda c: "Aumenta el riesgo" if c > 0 else "Reduce el riesgo")
    fig = px.bar(coef, x="Coeficiente", y="Variable", orientation="h", color="Efecto",
                 color_discrete_map={"Aumenta el riesgo": "#D1495B", "Reduce el riesgo": "#4C9F70"},
                 title="Regresión logística: 12 coeficientes más grandes")
    col_a.plotly_chart(fig, use_container_width=True)

    nombres_arbol = modelo_arbol.named_steps["prep"].get_feature_names_out()
    imp = pd.DataFrame({"Variable": [n.split("__", 1)[1] for n in nombres_arbol],
                        "Importancia": modelo_arbol.named_steps["modelo"].feature_importances_})
    imp = imp[imp["Importancia"] > 0].sort_values("Importancia")
    fig = px.bar(imp, x="Importancia", y="Variable", orientation="h",
                 color_discrete_sequence=["#30638E"], title="Árbol de decisión: importancia de variables")
    fig.update_xaxes(tickformat=".0%")
    col_b.plotly_chart(fig, use_container_width=True)

    st.write("Ambos modelos coinciden: la **edad** es el factor dominante, seguida del **sexo** y las "
             "medidas de obesidad (**cintura** e **IMC**). Que dos modelos con lógicas distintas lleguen a la "
             "misma conclusión indica que aprendieron patrones reales y no ruido.")

    with st.expander("🌳 Ver el diagrama del árbol de decisión (primeros 3 niveles)"):
        st.image("arbol_decision.png", use_container_width=True)
        st.caption("Los cortes aparecen en unidades estandarizadas por el escalado del Pipeline. "
                   "Por ejemplo, `edad <= -0.29` equivale a unos 43 años.")


# ---------------------------------------------------------------------------
# Navegación entre páginas
# ---------------------------------------------------------------------------
navegacion = st.navigation([
    st.Page(pagina_prediccion, title="Predicción", icon="🩺", default=True),
    st.Page(pagina_datos, title="Los datos", icon="📊"),
    st.Page(pagina_evaluacion, title="Entrenamiento y evaluación", icon="🧪"),
])
navegacion.run()
