import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Propagación inicial del COVID-19 por país")

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("COVID19_line_list_data.csv")
    return df

df = load_data()

# Convertir columnas de fecha
df["reporting date"] = pd.to_datetime(df["reporting date"], errors="coerce")
df["hosp_visit_date"] = pd.to_datetime(df["hosp_visit_date"], errors="coerce")

# Crear columna de fecha de inicio estimada
# Usamos hosp_visit_date como aproximación si no existe onset
df["onset_date"] = df["hosp_visit_date"]

# Eliminar filas sin fecha
df = df.dropna(subset=["onset_date"])

# Agrupar por país y fecha
df_grouped = (
    df.groupby(["country", "onset_date"])
      .size()
      .reset_index(name="cases")
)

# Ordenar
df_grouped = df_grouped.sort_values(["country", "onset_date"])

# Crear acumulado por país
df_grouped["cumulative_cases"] = (
    df_grouped.groupby("country")["cases"]
    .cumsum()
)

# Selector de países
countries = st.multiselect(
    "Selecciona países",
    df_grouped["country"].unique(),
    default=df_grouped["country"].unique()[:5]
)

df_filtered = df_grouped[df_grouped["country"].isin(countries)]

# Gráfica interactiva
fig = px.line(
    df_filtered,
    x="onset_date",
    y="cumulative_cases",
    color="country",
    markers=True,
    title="Inicio y propagación temprana de síntomas por país"
)

fig.update_layout(
    xaxis_title="Fecha de inicio de síntomas",
    yaxis_title="Casos acumulados",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)
