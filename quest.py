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
df["symptom_onset"] = pd.to_datetime(df["symptom_onset"], errors="coerce")
df = df.dropna(subset=["symptom_onset"])

# Normalizar columnas de Wuhan (convertir a numérico, NaN para no registrado)
df["visiting Wuhan"] = pd.to_numeric(df["visiting Wuhan"], errors="coerce")
df["from Wuhan"]     = pd.to_numeric(df["from Wuhan"],     errors="coerce")

# Crear columna de categoría Wuhan
def classify_wuhan(row):
    if row["from Wuhan"] == 1:
        return "De Wuhan"
    elif row["visiting Wuhan"] == 1:
        return "Visitó Wuhan"
    elif row["from Wuhan"] == 0 and row["visiting Wuhan"] == 0:
        return "Sin vínculo con Wuhan"
    else:
        return "No registrado"

df["wuhan_status"] = df.apply(classify_wuhan, axis=1)

# -------------------------
# Agrupación base
# -------------------------
df_grouped = (
    df.groupby(["country", "symptom_onset", "wuhan_status"])
      .size()
      .reset_index(name="cases")
      .sort_values(["country", "symptom_onset"])
)

df_grouped["cumulative_cases"] = (
    df_grouped.groupby(["country", "wuhan_status"])["cases"].cumsum()
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
        df_grouped["symptom_onset"].min(),
        df_grouped["symptom_onset"].max()
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

# --- Filtro Wuhan ---
st.sidebar.markdown("---")
st.sidebar.subheader("Vínculo con Wuhan")

wuhan_mode = st.sidebar.radio(
    "Modo de visualización Wuhan",
    ["Por categoría", "Total sin distinción"],
    help=(
        "Por categoría: filtra y/o colorea según el vínculo con Wuhan.\n\n"
        "Total sin distinción: suma todos los casos por país ignorando el vínculo."
    )
)

wuhan_options = ["De Wuhan", "Visitó Wuhan", "Sin vínculo con Wuhan", "No registrado"]

if wuhan_mode == "Por categoría":
    wuhan_selection = st.sidebar.multiselect(
        "Mostrar categorías",
        options=wuhan_options,
        default=wuhan_options
    )
    wuhan_color = st.sidebar.checkbox(
        "Colorear por vínculo con Wuhan (en lugar de por país)",
        value=False
    )
else:
    wuhan_selection = wuhan_options   # incluir todo
    wuhan_color = False               # sin distinción de color por Wuhan

# -------------------------
# Aplicar filtros
# -------------------------
df_filtered = df_grouped[
    (df_grouped["country"].isin(countries)) &
    (df_grouped["wuhan_status"].isin(wuhan_selection))
]

if len(date_range) == 2:
    df_filtered = df_filtered[
        (df_filtered["symptom_onset"] >= pd.to_datetime(date_range[0])) &
        (df_filtered["symptom_onset"] <= pd.to_datetime(date_range[1]))
    ]

# Comparación por primeros N días
if days_from_first > 0:
    df_filtered = df_filtered.copy()
    df_filtered["days_since_first"] = (
        df_filtered["symptom_onset"] -
        df_filtered.groupby("country")["symptom_onset"].transform("min")
    ).dt.days
    df_filtered = df_filtered[df_filtered["days_since_first"] <= days_from_first]
    x_axis = "days_since_first"
    x_label = "Días desde primer caso"
else:
    x_axis = "symptom_onset"
    x_label = "Fecha"

# Selección variable Y
y_axis = "cumulative_cases" if case_type == "Casos acumulados" else "cases"

# -------------------------
# Agrupación final según modo de color
# -------------------------

# Modo "Total sin distinción": colapsar wuhan_status y reagrupar por país
if wuhan_mode == "Total sin distinción":
    agg_x = x_axis if x_axis == "days_since_first" else "symptom_onset"
    df_plot = (
        df_filtered
        .groupby(["country", agg_x])["cases"]
        .sum()
        .reset_index()
        .sort_values(["country", agg_x])
    )
    df_plot["cumulative_cases"] = df_plot.groupby("country")["cases"].cumsum()
    y_axis_plot = "cumulative_cases" if case_type == "Casos acumulados" else "cases"

    fig = px.line(
        df_plot,
        x=agg_x,
        y=y_axis_plot,
        color="country",
        markers=True,
        template="plotly_white"
    )
    legend_title = "País"

elif wuhan_color:
    # Agrupar por wuhan_status (sumar todos los países seleccionados)
    agg_cols = [x_axis, "wuhan_status"] if x_axis == "days_since_first" else ["symptom_onset", "wuhan_status"]
    df_plot = (
        df_filtered.groupby(agg_cols)[y_axis]
        .sum()
        .reset_index()
    )
    color_col = "wuhan_status"
    legend_title = "Vínculo con Wuhan"

    # Paleta fija por categoría
    color_map = {
        "De Wuhan":             "#e74c3c",
        "Visitó Wuhan":         "#e67e22",
        "Sin vínculo con Wuhan":"#2ecc71",
        "No registrado":        "#95a5a6"
    }
    fig = px.line(
        df_plot,
        x=x_axis,
        y=y_axis,
        color=color_col,
        color_discrete_map=color_map,
        markers=True,
        template="plotly_white"
    )
else:
    # Agrupación por país + wuhan_status → línea punteada/sólida según vínculo
    df_plot = df_filtered.copy()
    df_plot["serie"] = df_plot["country"] + " — " + df_plot["wuhan_status"]

    fig = px.line(
        df_plot,
        x=x_axis,
        y=y_axis,
        color="country",
        line_dash="wuhan_status",
        markers=True,
        template="plotly_white",
        hover_data={"wuhan_status": True}
    )
    legend_title = "País / Vínculo Wuhan"

# -------------------------
# Escala y layout
# -------------------------
if scale_type == "Logarítmica":
    fig.update_yaxes(type="log")

fig.update_layout(
    title="Propagación del COVID-19 según inicio de síntomas",
    xaxis_title=x_label,
    yaxis_title=case_type,
    hovermode="x unified",
    legend_title=legend_title,
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Tabla resumen Wuhan
# -------------------------
with st.expander("📊 Distribución por vínculo con Wuhan"):
    summary = (
        df[df["country"].isin(countries)]
        .groupby("wuhan_status")
        .size()
        .reset_index(name="total_casos")
        .sort_values("total_casos", ascending=False)
    )
    st.dataframe(summary, use_container_width=True)
