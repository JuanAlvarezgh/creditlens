import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Make the project root importable regardless of where streamlit was launched
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="CreditLens",
    page_icon=":material/credit_score:",
    layout="wide",
)
st.title("CreditLens — Análisis de Riesgo Crediticio")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Resumen Ejecutivo",
        "Exploración de Datos",
        "Rendimiento del Modelo",
        "Scoring Manual",
        "Acerca del Proyecto",
    ]
)


@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    import psycopg2

    conn = psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        dbname=settings.postgres_db,
    )
    try:
        return pd.read_sql_query("SELECT * FROM mart_credit_features LIMIT 10000", conn)
    finally:
        conn.close()


ALGO_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
    "xgboost_deep": "XGBoost (Deep)",
    "lightgbm": "LightGBM",
    "lightgbm_tuned": "LightGBM (Tuned)",
}


@st.cache_data(ttl=60)
def get_algorithm_comparison() -> pd.DataFrame:
    """Fetch the latest run per algorithm from MLflow, sorted by AUC-ROC desc."""
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("credit_risk")
    if not experiment:
        return pd.DataFrame()

    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    seen = set()
    rows = []
    for run in runs:
        algo = run.data.tags.get("mlflow.runName", "unknown")
        if algo in seen:
            continue
        seen.add(algo)
        rows.append(
            {
                "Algoritmo": ALGO_DISPLAY.get(algo, algo),
                "AUC-ROC": run.data.metrics.get("auc_roc"),
                "KS Statistic": run.data.metrics.get("ks_statistic"),
                "Gini": run.data.metrics.get("gini"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
    return df


# Glossary of feature descriptions — reused across tabs
FEATURE_HELP = {
    "revolving_utilization": (
        "Porcentaje del límite total de tarjetas y líneas rotativas que el "
        "cliente está usando. Valores altos (>70%) son una señal fuerte de "
        "estrés financiero."
    ),
    "age": "Edad del solicitante en años. Clientes más jóvenes suelen tener historial crediticio más corto.",
    "debt_ratio": (
        "Total mensual de pagos de deuda dividido entre el ingreso mensual. "
        "Valores >1 significan que el cliente paga más en deudas que lo que "
        "gana — alto riesgo."
    ),
    "monthly_income": "Ingreso bruto mensual del solicitante en USD.",
    "total_late_payments": (
        "Suma de todas las moras (30-59 + 60-89 + 90+ días) en los últimos "
        "2 años. Indicador directo de historial de incumplimiento."
    ),
    "dependents": "Personas económicamente a cargo del solicitante (hijos, padres mayores, etc.).",
    "open_credit_lines": "Número total de tarjetas de crédito y préstamos activos.",
    "times_90_days_late": "Veces que el cliente se atrasó 90+ días en pagos. El peor tipo de mora.",
    "real_estate_loans": "Hipotecas y préstamos sobre bienes raíces activos.",
}


# ── Tab 1: Resumen Ejecutivo ──────────────────────────────────────────────────
with tab1:
    st.header("Resumen Ejecutivo")
    st.markdown(
        "Vista de alto nivel sobre el dataset cargado en PostgreSQL "
        "(`mart_credit_features`). Muestra qué proporción de la población "
        "histórica entró en mora y cómo se distribuye el riesgo."
    )

    df = load_data()
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Total Solicitudes",
        f"{len(df):,}",
        help="Número de solicitudes de crédito en la capa analítica (mart).",
    )
    default_rate = df["label"].mean()
    col2.metric(
        "Tasa de Default Histórica",
        f"{default_rate:.1%}",
        help=(
            "Porcentaje de clientes que defaultearon (mora grave de 90+ días "
            "en los siguientes 2 años). Es la verdad histórica que el modelo "
            "aprende a predecir."
        ),
    )
    col3.metric(
        "Aprobación Estimada",
        f"{1 - default_rate:.1%}",
        help=(
            "Complemento de la tasa de default: porcentaje que NO defaulteó "
            "y representa candidatos viables para aprobación."
        ),
    )

    st.divider()
    st.subheader("Distribución por Segmento de Utilización de Crédito")
    st.caption(
        "Los segmentos se definen sobre `revolving_utilization`: "
        "low (≤30%), medium (30-70%), high (>70%). "
        "Mayor proporción en `high` suele correlacionar con mayor riesgo."
    )
    fig = px.pie(
        df["utilization_segment"].value_counts().reset_index(),
        names="utilization_segment",
        values="count",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Exploración de Datos ───────────────────────────────────────────────
with tab2:
    st.header("Exploración de Datos")
    st.markdown(
        "Análisis exploratorio para entender qué variables del cliente "
        "más impactan en el riesgo. Si la distribución de una feature se "
        "ve distinta entre clientes que defaultearon (1) vs los que no (0), "
        "esa variable tiene poder predictivo."
    )

    df = load_data()
    feature = st.selectbox(
        "Selecciona una feature para analizar",
        [
            "revolving_utilization",
            "age",
            "debt_ratio",
            "monthly_income",
            "total_late_payments",
            "dependents",
        ],
        help="Elige una variable del solicitante para ver cómo se relaciona con el default.",
    )

    if feature in FEATURE_HELP:
        st.info(f"**{feature}** — {FEATURE_HELP[feature]}")

    fig = px.histogram(
        df,
        x=feature,
        color=df["label"].astype(str),
        barmode="overlay",
        title=f"Distribución de '{feature}' separada por Default",
        labels={"color": "Default (1=Sí, 0=No)"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Si las dos curvas (rojo = defaulteó, azul = no) están separadas, "
        "esta feature ayuda al modelo a discriminar."
    )

    st.divider()
    st.subheader("Matriz de Correlación")
    st.caption(
        "Heatmap de correlación entre las features principales y el `label` "
        "(default). Valores cercanos a +1 o -1 significan correlación "
        "fuerte; cercanos a 0 significan ausencia de relación lineal."
    )
    corr_cols = [
        "revolving_utilization",
        "age",
        "debt_ratio",
        "monthly_income",
        "total_late_payments",
        "dependents",
        "label",
    ]
    fig2 = px.imshow(df[corr_cols].corr(), text_auto=".2f")
    st.plotly_chart(fig2, use_container_width=True)


# ── Tab 3: Rendimiento del Modelo ─────────────────────────────────────────────
with tab3:
    st.header("Rendimiento del Modelo")
    st.markdown(
        "Información en vivo del modelo en producción (cargado desde "
        "MLflow Model Registry). Si la API no está corriendo, esta vista "
        "no podrá conectarse."
    )

    try:
        resp = requests.get(f"{API_URL}/api/v1/model/info", timeout=5)
        if resp.status_code == 200:
            info = resp.json()

            st.subheader("Modelo en Producción")
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Algoritmo",
                info.get("algorithm") or "—",
                help=(
                    "Familia de algoritmo entrenado. Posibles: "
                    "`LogisticRegression` (lineal interpretable), "
                    "`XGBClassifier` (gradient boosting), "
                    "`LGBMClassifier` (LightGBM, rápido y preciso)."
                ),
            )
            col2.metric(
                "Versión",
                info["version"],
                help="Versión en el MLflow Model Registry. Cada reentrenamiento crea una nueva versión.",
            )
            col3.metric(
                "Stage",
                info["stage"],
                help="Etapa del modelo: Production (sirviendo predicciones), Staging, Archived.",
            )

            st.divider()
            st.subheader("Métricas Financieras (test set)")
            mcol1, mcol2, mcol3 = st.columns(3)

            auc = info.get("auc_roc")
            mcol1.metric(
                "AUC-ROC",
                f"{auc:.4f}" if auc is not None else "—",
                help=(
                    "Área bajo la curva ROC. 0.5 = aleatorio (sin valor), "
                    "1.0 = perfecto. En riesgo crediticio: 0.60-0.70 "
                    "aceptable, 0.70-0.80 bueno, >0.80 excelente."
                ),
            )

            ks = info.get("ks_statistic")
            mcol2.metric(
                "KS Statistic",
                f"{ks:.4f}" if ks is not None else "—",
                help=(
                    "Kolmogorov-Smirnov: máxima separación entre la distribución "
                    "de scores de buenos y malos pagadores. >0.30 se "
                    "considera bueno en banca. Métrica estándar en credit scoring."
                ),
            )

            gini = info.get("gini")
            mcol3.metric(
                "Gini Coefficient",
                f"{gini:.4f}" if gini is not None else "—",
                help=(
                    "Coeficiente de Gini = 2 × AUC - 1. Métrica preferida "
                    "por reguladores financieros. Va de 0 (aleatorio) a 1 (perfecto)."
                ),
            )

            st.divider()
            st.caption(
                f"Nombre interno: `{info['model_name']}` · " f"Registrado: {info['registered_at']}"
            )
        else:
            st.warning(f"No se pudo obtener info del modelo (HTTP {resp.status_code}).")
    except Exception as e:
        st.warning(
            "API no disponible. Inicia la API primero con:\n\n"
            "```bash\nuvicorn api.main:app --port 8000\n```\n\n"
            f"_Detalle del error: {e}_"
        )

    # ── Comparison of all algorithms trained ──────────────────────────────────
    st.divider()
    st.subheader("Algoritmos Entrenados")
    st.caption(
        "Comparación de todos los modelos evaluados durante el experimento. "
        "El modelo con mayor AUC-ROC se promueve automáticamente a Production "
        "via MLflow Model Registry."
    )

    try:
        comp_df = get_algorithm_comparison()
        if comp_df.empty:
            st.info("No hay experimentos registrados en MLflow todavía.")
        else:
            st.dataframe(
                comp_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Algoritmo": st.column_config.TextColumn("Algoritmo", width="medium"),
                    "AUC-ROC": st.column_config.NumberColumn(
                        "AUC-ROC", format="%.4f", width="small"
                    ),
                    "KS Statistic": st.column_config.NumberColumn(
                        "KS Statistic", format="%.4f", width="small"
                    ),
                    "Gini": st.column_config.NumberColumn("Gini", format="%.4f", width="small"),
                },
            )
            st.caption(
                "Familias de modelos comparadas: regresión lineal (interpretable), "
                "gradient boosting clásico (XGBoost) y gradient boosting eficiente "
                "(LightGBM, desarrollado por Microsoft). Los modelos de árboles "
                "suelen capturar mejor las interacciones no lineales entre features."
            )
    except Exception as e:
        st.info(f"No se pudo conectar al MLflow Tracking Server: {e}")


# ── Tab 4: Scoring Manual ─────────────────────────────────────────────────────
with tab4:
    st.header("Scoring Manual")
    st.markdown(
        "Evalúa una solicitud de crédito en tiempo real. Ingresa los datos del "
        "solicitante y el modelo devolverá la probabilidad de default, "
        "el nivel de riesgo y las 3 variables que más influyeron "
        "en la decisión (explicabilidad SHAP)."
    )

    with st.expander("¿Qué significa cada campo?", expanded=False):
        st.markdown(
            "- **Edad** — edad del solicitante (18-100 años).\n"
            "- **Ingreso mensual** — ingreso bruto en USD por mes.\n"
            f"- **Ratio de deuda** — {FEATURE_HELP['debt_ratio']}\n"
            f"- **Utilización crédito rotativo** — {FEATURE_HELP['revolving_utilization']}\n"
            f"- **Dependientes** — {FEATURE_HELP['dependents']}\n"
            "- **Moras 30-59 / 60-89 / 90+ días** — número de veces que el "
            "cliente se atrasó esos rangos de días en los últimos 2 años. "
            "Las moras de 90+ son las más graves.\n"
            f"- **Líneas de crédito abiertas** — {FEATURE_HELP['open_credit_lines']}\n"
            f"- **Préstamos hipotecarios** — {FEATURE_HELP['real_estate_loans']}"
        )

    with st.form("score_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider(
                "Edad",
                18,
                100,
                35,
                help="Edad del solicitante en años.",
            )
            monthly_income = st.number_input(
                "Ingreso mensual (USD)",
                100.0,
                100000.0,
                5000.0,
                help="Ingreso bruto mensual del solicitante.",
            )
            debt_ratio = st.slider(
                "Ratio de deuda",
                0.0,
                2.0,
                0.25,
                help=FEATURE_HELP["debt_ratio"],
            )
            revolving_utilization = st.slider(
                "Utilización crédito rotativo",
                0.0,
                1.0,
                0.45,
                help=FEATURE_HELP["revolving_utilization"],
            )
            dependents = st.number_input(
                "Dependientes",
                0,
                20,
                2,
                help=FEATURE_HELP["dependents"],
            )
        with col2:
            times_30_59 = st.number_input(
                "Moras 30-59 días",
                0,
                20,
                0,
                help="Veces atrasado 30-59 días en los últimos 2 años. Indicador leve.",
            )
            times_60_89 = st.number_input(
                "Moras 60-89 días",
                0,
                20,
                0,
                help="Veces atrasado 60-89 días. Indicador moderado.",
            )
            times_90 = st.number_input(
                "Moras 90+ días",
                0,
                20,
                0,
                help=FEATURE_HELP["times_90_days_late"],
            )
            open_credit_lines = st.number_input(
                "Líneas de crédito abiertas",
                0,
                50,
                4,
                help=FEATURE_HELP["open_credit_lines"],
            )
            real_estate_loans = st.number_input(
                "Préstamos hipotecarios",
                0,
                20,
                1,
                help=FEATURE_HELP["real_estate_loans"],
            )
        submitted = st.form_submit_button("Evaluar Riesgo")

    if submitted:
        payload = {
            "revolving_utilization": revolving_utilization,
            "age": age,
            "times_30_59_days_late": int(times_30_59),
            "debt_ratio": debt_ratio,
            "monthly_income": monthly_income,
            "open_credit_lines": int(open_credit_lines),
            "times_90_days_late": int(times_90),
            "real_estate_loans": int(real_estate_loans),
            "times_60_89_days_late": int(times_60_89),
            "dependents": int(dependents),
        }
        try:
            with st.spinner("Evaluando riesgo crediticio..."):
                resp = requests.post(f"{API_URL}/api/v1/score", json=payload, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                risk = result["risk_level"]
                prob = result["probability_of_default"]
                color = {"Bajo": "green", "Medio": "orange", "Alto": "red"}[risk]

                st.divider()
                st.markdown(f"### Resultado: :{color}[{risk}]")
                rcol1, rcol2 = st.columns(2)
                rcol1.metric(
                    "Probabilidad de Default",
                    f"{prob:.1%}",
                    help=(
                        "Probabilidad estimada de que este cliente entre en mora "
                        "grave (90+ días) en los próximos 2 años."
                    ),
                )
                rcol2.metric(
                    "Umbrales del Modelo",
                    "Bajo <30% · Medio 30-60% · Alto >60%",
                )

                st.subheader("Factores más influyentes (SHAP)")
                st.caption(
                    "Las 3 variables que más empujaron la decisión. ↑ aumenta "
                    "el riesgo, ↓ reduce el riesgo. Esto es lo que un banco "
                    "regulado usaría para explicarle al cliente por qué fue "
                    "aprobado o rechazado."
                )
                for item in result["shap_explanation"]:
                    direction = "↑ aumenta" if item["impact"] > 0 else "↓ reduce"
                    feature_name = item["feature"]
                    description = FEATURE_HELP.get(feature_name, "")
                    st.write(
                        f"**{feature_name}**: {direction} el riesgo " f"({item['impact']:+.3f})"
                    )
                    if description:
                        st.caption(description)
            else:
                st.error(f"Error de la API: {resp.status_code} — {resp.text}")
        except Exception as e:
            st.error(f"No se pudo conectar a la API: {e}")

    # ── Worked examples ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Casos de Ejemplo")
    st.markdown(
        "Dos perfiles realistas que ilustran cómo cada variable empuja la "
        "decisión del modelo. Puedes copiar estos valores en el formulario "
        "de arriba para ver el resultado en vivo."
    )

    ex_col1, ex_col2 = st.columns(2)

    with ex_col1:
        st.markdown("#### Caso A — Profesional Establecido (Bajo Riesgo)")
        st.markdown(
            "María, 42 años, Senior Software Engineer. Hipoteca al día, "
            "tarjeta de crédito usada con disciplina, trayectoria laboral "
            "estable."
        )
        st.markdown(
            """
| Variable | Valor | Lectura |
|---|---|---|
| Edad | 42 | Edad madura — mayor estabilidad |
| Ingreso mensual | $6,000 USD | Profesional con experiencia |
| Ratio de deuda | 0.22 | Paga 22% del ingreso en deudas — zona sana (<30%) |
| Utilización rotativo | 0.18 | Solo 18% del cupo — disciplina financiera |
| Dependientes | 1 | Carga manejable |
| Moras 30-59 | 0 | Sin historial de retrasos |
| Moras 60-89 | 0 | Sin historial de retrasos |
| Moras 90+ | 0 | Sin defaults previos |
| Líneas abiertas | 6 | Mix saludable de productos |
| Hipotecas | 1 | Hipoteca pagada al día |
            """
        )
        st.success("**Predicción esperada:** Bajo riesgo · ~10-18% probabilidad de default")

    with ex_col2:
        st.markdown("#### Caso B — Joven en Estrés Financiero (Alto Riesgo)")
        st.markdown(
            "Carlos, 24 años, primer empleo formal. Usó tarjetas sin medida "
            "en la universidad, tiene moras recientes, ingreso bajo, "
            "personas a cargo."
        )
        st.markdown(
            """
| Variable | Valor | Lectura |
|---|---|---|
| Edad | 24 | Historial crediticio corto, poca estabilidad |
| Ingreso mensual | $1,200 USD | Salario inicial |
| Ratio de deuda | 1.50 | **Paga 50% MÁS de lo que gana en deudas — insostenible** |
| Utilización rotativo | 0.92 | Tarjeta al 92% del cupo — estrés grave |
| Dependientes | 2 | Personas a cargo agravan el riesgo |
| Moras 30-59 | 4 | Patrón repetido de retrasos cortos |
| Moras 60-89 | 2 | Empeoramiento del incumplimiento |
| Moras 90+ | 1 | **Ya tuvo un default — fuerte predictor** |
| Líneas abiertas | 10 | Demasiados productos para su capacidad |
| Hipotecas | 0 | Sin activos respaldando |
            """
        )
        st.error("**Predicción esperada:** Alto riesgo · >85% probabilidad de default")


# ── Tab 5: Acerca del Proyecto ────────────────────────────────────────────────
with tab5:
    st.header("Acerca del Proyecto")
    st.markdown(
        "CreditLens es un sistema end-to-end de análisis de riesgo crediticio "
        "que cubre el ciclo completo del dato: ingesta en streaming, "
        "transformaciones declarativas, entrenamiento y registro de modelos, "
        "scoring en tiempo real y visualización analítica. Diseñado para "
        "demostrar competencias de Data Engineering y Machine Learning "
        "aplicadas a un caso real de la industria financiera."
    )

    st.divider()
    st.subheader("Tecnologías Utilizadas")
    st.caption("Stack completo del proyecto — cada herramienta cubre una capa del pipeline.")

    tech_stack = [
        (
            "Python",
            "https://cdn.simpleicons.org/python",
            "Lenguaje principal del backend, ML y orquestación.",
        ),
        (
            "Apache Kafka",
            "https://cdn.simpleicons.org/apachekafka",
            "Streaming de solicitudes en tiempo real.",
        ),
        (
            "PostgreSQL",
            "https://cdn.simpleicons.org/postgresql",
            "Capa raw — almacén de datos crudos.",
        ),
        (
            "dbt",
            "https://api.iconify.design/logos:dbt-icon.svg",
            "Transformaciones SQL declarativas (raw → staging → mart).",
        ),
        (
            "Apache Airflow",
            "https://cdn.simpleicons.org/apacheairflow",
            "Orquestación de pipelines: ingesta, dbt, reentrenamiento.",
        ),
        ("MLflow", "https://cdn.simpleicons.org/mlflow", "Experiment tracking y Model Registry."),
        (
            "scikit-learn",
            "https://cdn.simpleicons.org/scikitlearn",
            "Modelo base (Logistic Regression) y métricas.",
        ),
        ("XGBoost", None, "Gradient boosting de árboles para clasificación."),
        ("LightGBM", None, "Gradient boosting eficiente desarrollado por Microsoft."),
        ("SHAP", None, "Explicabilidad de modelos — Shapley additive values."),
        ("FastAPI", "https://cdn.simpleicons.org/fastapi", "API REST de scoring en tiempo real."),
        ("Pydantic", "https://cdn.simpleicons.org/pydantic", "Validación de esquemas tipados."),
        ("Streamlit", "https://cdn.simpleicons.org/streamlit", "Este dashboard analítico."),
        ("Plotly", "https://cdn.simpleicons.org/plotly", "Visualizaciones interactivas."),
        ("Docker", "https://cdn.simpleicons.org/docker", "Contenerización de todos los servicios."),
        (
            "GitHub Actions",
            "https://cdn.simpleicons.org/githubactions",
            "CI/CD: lint y tests automáticos en cada push.",
        ),
    ]

    # 4-column grid
    for i in range(0, len(tech_stack), 4):
        cols = st.columns(4)
        for j, (name, logo, desc) in enumerate(tech_stack[i : i + 4]):
            with cols[j]:
                if logo:
                    st.image(logo, width=48)
                else:
                    st.markdown("&nbsp;", unsafe_allow_html=True)
                st.markdown(f"**{name}**")
                st.caption(desc)

    st.divider()
    st.subheader("Sobre el Creador")

    creator_col1, creator_col2 = st.columns([1, 3])
    with creator_col1:
        st.image(
            "https://github.com/JuanAlvarezgh.png",
            width=160,
        )
    with creator_col2:
        st.markdown(
            """
            **Juan Alvarez**

            Estudiante de Ingeniería de Datos y Software en la
            **Universidad de San Buenaventura**.

            Líneas de trabajo: arquitectura de datos end-to-end, machine
            learning aplicado al sector financiero y APIs de producción.
            Familiaridad con stacks modernos (Kafka, dbt, MLflow, FastAPI)
            y métricas regulatorias estándar (KS, Gini, AUC-ROC).

            **Áreas de interés:** scoring crediticio, MLOps,
            explicabilidad de modelos (SHAP) y monitoreo de drift en
            producción.
            """
        )

    st.divider()
    st.subheader("Contacto")

    contact_col1, contact_col2, contact_col3 = st.columns(3)
    with contact_col1:
        st.image("https://cdn.simpleicons.org/github", width=32)
        st.markdown("**GitHub**")
        st.markdown("[github.com/JuanAlvarezgh](https://github.com/JuanAlvarezgh)")
    with contact_col2:
        st.image("https://cdn.simpleicons.org/gmail", width=32)
        st.markdown("**Email**")
        st.markdown("[juanalvarezghcode@gmail.com](mailto:juanalvarezghcode@gmail.com)")
    with contact_col3:
        st.image("https://api.iconify.design/logos:linkedin-icon.svg", width=32)
        st.markdown("**LinkedIn**")
        st.markdown("[linkedin.com/in/juanalvarezgh]" "(https://www.linkedin.com/in/juanalvarezgh)")

    st.divider()
    st.caption(
        "Código fuente: [github.com/JuanAlvarezgh/creditlens](https://github.com/JuanAlvarezgh/creditlens) · "
        "Licencia MIT"
    )
