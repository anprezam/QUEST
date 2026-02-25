import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Análisis interactivo de propagación inicial del COVID-19")

# -------------------------
# Cargar datos
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("COVID19_line_list_data.csv")
    return df

df = load_data()

# -------------------------
# Limpieza y preparación
# -------------------------
df["hosp_visit_date"] = pd.to_datetime(df["hosp_visit_date"], errors="coerce")
df = df.dropna(subset=["hosp_visit_date"])

df_grouped = (
    df.groupby(["country", "hosp_visit_date"])
      .size()
      .reset_index(name="cases")
      .sort_values(["country", "hosp_visit_date"])
)

df_grouped["cumulative_cases"] = (
    df_grouped.groupby("country")["cases"].cumsum()
)

# -------------------------
# Panel lateral interactivo
# -------------------------
st.sidebar.header("Filtros")

countries = st.sidebar.multiselect(
    "Selecciona países",
    df_grouped["country"].unique(),
    default=df_grouped["country"].unique()[:5]
)

date_range = st.sidebar.date_input(
    "Selecciona rango de fechas",
    [
        df_grouped["hosp_visit_date"].min(),
        df_grouped["hosp_visit_date"].max()
    ]
)

days_from_first = st.sidebar.slider(
    "Mostrar primeros N días desde primer caso",
    min_value=0,
    max_value=120,
    value=0
)

scale_type = st.sidebar.radio(
    "Tipo de escala",
    ["Lineal", "Logarítmica"]
)

case_type = st.sidebar.radio(
    "Tipo de visualización",
    ["Casos acumulados", "Casos diarios"]
)

# -------------------------
# Aplicar filtros
# -------------------------
df_filtered = df_grouped[df_grouped["country"].isin(countries)]

if len(date_range) == 2:
    df_filtered = df_filtered[
        (df_filtered["hosp_visit_date"] >= pd.to_datetime(date_range[0])) &
        (df_filtered["hosp_visit_date"] <= pd.to_datetime(date_range[1]))
    ]

# Comparación por primeros N días
if days_from_first > 0:
    df_filtered["days_since_first"] = (
        df_filtered["hosp_visit_date"] -
        df_filtered.groupby("country")["hosp_visit_date"].transform("min")
    ).dt.days
    
    df_filtered = df_filtered[df_filtered["days_since_first"] <= days_from_first]
    x_axis = "days_since_first"
    x_label = "Días desde primer caso"
else:
    x_axis = "hosp_visit_date"
    x_label = "Fecha"

# Selección variable Y
if case_type == "Casos acumulados":
    y_axis = "cumulative_cases"
else:
    y_axis = "cases"

# -------------------------
# Gráfica
# -------------------------
fig = px.line(
    df_filtered,
    x=x_axis,
    y=y_axis,
    color="country",
    markers=True,
    template="plotly_white"
)

if scale_type == "Logarítmica":
    fig.update_yaxes(type="log")

fig.update_layout(
    title="Propagación del COVID-19 según inicio de síntomas",
    xaxis_title=x_label,
    yaxis_title=case_type,
    hovermode="x unified",
    legend_title="País",
    height=600
)

st.plotly_chart(fig, use_container_width=True)
