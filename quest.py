import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="COVID-19 · Propagación Global",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}
.stApp { background-color: #0a0a0f; }

h1, h2, h3 { font-family: 'Space Mono', monospace; color: #ff6b35; }

.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #ff6b35;
    border-radius: 8px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 0 20px rgba(255,107,53,0.15);
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #ff6b35;
}
.metric-label {
    font-size: 0.8rem;
    color: #8888aa;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 4px;
}

.country-pill {
    display: inline-block;
    background: #ff6b35;
    color: #0a0a0f;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 700;
    margin: 2px;
    font-family: 'Space Mono', monospace;
}

.stSidebar { background-color: #0f0f1a !important; border-right: 1px solid #ff6b35; }
.stSlider > div > div { background-color: #ff6b35 !important; }
.stMultiSelect > div { background-color: #1a1a2e !important; border-color: #ff6b35 !important; }

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #8888aa;
    text-transform: uppercase;
    letter-spacing: 3px;
    border-bottom: 1px solid #ff6b35;
    padding-bottom: 4px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─── DATA LOADER ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Demo data parecido a la estructura real
        np.random.seed(42)
        countries = {
            "China": ("2019-11-01", "2019-12-20", 450),
            "Thailand": ("2020-01-08", "2020-01-25", 30),
            "Japan": ("2020-01-10", "2020-02-10", 80),
            "South Korea": ("2020-01-15", "2020-02-20", 200),
            "USA": ("2020-01-19", "2020-03-15", 300),
            "Germany": ("2020-01-24", "2020-03-10", 150),
            "France": ("2020-01-24", "2020-03-20", 120),
            "Singapore": ("2020-01-23", "2020-02-15", 60),
            "Australia": ("2020-01-25", "2020-02-28", 45),
            "Vietnam": ("2020-01-22", "2020-02-20", 40),
            "Canada": ("2020-01-25", "2020-03-10", 90),
            "UK": ("2020-01-28", "2020-03-20", 110),
            "Italy": ("2020-01-31", "2020-03-25", 280),
            "Spain": ("2020-01-31", "2020-03-28", 200),
            "Iran": ("2020-02-19", "2020-03-20", 150),
            "India": ("2020-01-30", "2020-03-20", 80),
            "Brazil": ("2020-02-25", "2020-03-30", 100),
        }
        rows = []
        for country, (start, end, n) in countries.items():
            dates = pd.date_range(start, end, periods=n)
            for d in dates:
                rows.append({
                    "country": country,
                    "symptom_onset": d,
                    "gender": np.random.choice(["male", "female"], p=[0.55, 0.45]),
                    "age": np.random.randint(15, 85),
                    "death": np.random.choice([0, 1], p=[0.97, 0.03]),
                    "recovered": np.random.choice([0, 1], p=[0.3, 0.7]),
                    "visiting_wuhan": np.random.choice([0, 1], p=[0.8, 0.2]),
                })
        df = pd.DataFrame(rows)
    return df

# ─── PREPROCESSING ────────────────────────────────────────────────────────────
def preprocess(df):
    # Normalizar nombres de columnas
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    col_map = {
        "visiting_wuhan": "visiting wuhan",
        "from_wuhan": "from wuhan",
    }
    for new, old in col_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    
    date_col = next((c for c in df.columns if "onset" in c), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.rename(columns={date_col: "symptom_onset"})

    country_col = next((c for c in df.columns if "country" in c), "country")
    df = df.rename(columns={country_col: "country"})
    df["country"] = df["country"].astype(str).str.strip()

    return df.dropna(subset=["symptom_onset", "country"])

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦠 COVID-19")
    st.markdown("<div class='section-title'>Fuente de datos</div>", unsafe_allow_html=True)
    
    uploaded = st.file_uploader("Sube tu CSV", type=["csv", "tsv"])
    st.markdown("---")

    st.markdown("<div class='section-title'>Filtros</div>", unsafe_allow_html=True)

    raw = load_data(uploaded)
    df = preprocess(raw.copy())

    min_date = df["symptom_onset"].min().date()
    max_date = df["symptom_onset"].max().date()

    date_range = st.date_input(
        "Rango de fechas",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    all_countries = sorted(df["country"].unique())
    selected_countries = st.multiselect(
        "Países",
        options=all_countries,
        default=all_countries,
        help="Selecciona uno o más países"
    )

    view_mode = st.radio(
        "Vista de tiempo",
        ["Diario", "Semanal", "Mensual"],
        index=1,
        horizontal=True,
    )

    show_wuhan = st.checkbox("Resaltar casos de Wuhan", value=True)

# ─── FILTER DATA ──────────────────────────────────────────────────────────────
if len(date_range) == 2:
    start_d, end_d = date_range
    mask = (
        (df["symptom_onset"].dt.date >= start_d) &
        (df["symptom_onset"].dt.date <= end_d) &
        (df["country"].isin(selected_countries))
    )
    fdf = df[mask].copy()
else:
    fdf = df[df["country"].isin(selected_countries)].copy()

# Resample period
freq_map = {"Diario": "D", "Semanal": "W", "Mensual": "ME"}
freq = freq_map[view_mode]

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("# PROPAGACIÓN GLOBAL COVID-19")
st.markdown("<div class='section-title'>Análisis temporal de inicio de síntomas por país</div>", unsafe_allow_html=True)

# ─── KPI CARDS ────────────────────────────────────────────────────────────────
first_onset = fdf.groupby("country")["symptom_onset"].min().reset_index()
first_country = first_onset.loc[first_onset["symptom_onset"].idxmin(), "country"] if not first_onset.empty else "—"
first_date = first_onset["symptom_onset"].min().strftime("%d %b %Y") if not first_onset.empty else "—"

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{len(selected_countries)}</div>
        <div class='metric-label'>Países analizados</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{len(fdf):,}</div>
        <div class='metric-label'>Casos totales</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{first_country}</div>
        <div class='metric-label'>Primer reporte</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{first_date}</div>
        <div class='metric-label'>Fecha primer síntoma</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── CHART 1: Timeline — First Symptom per Country ────────────────────────────
st.markdown("### 📅 Línea de tiempo — Primer síntoma por país")
st.markdown("*Cuando cada país reportó su primer caso con inicio de síntomas*")

timeline_df = first_onset.sort_values("symptom_onset")
timeline_df["days_after_first"] = (
    timeline_df["symptom_onset"] - timeline_df["symptom_onset"].min()
).dt.days

# Count total cases per country for bubble size
cases_count = fdf.groupby("country").size().reset_index(name="cases")
timeline_df = timeline_df.merge(cases_count, on="country", how="left")

fig_timeline = px.scatter(
    timeline_df,
    x="symptom_onset",
    y="country",
    size="cases",
    color="days_after_first",
    color_continuous_scale=[[0, "#ff6b35"], [0.3, "#ff9f1c"], [0.7, "#e84545"], [1, "#6c63ff"]],
    size_max=45,
    hover_data={"cases": True, "days_after_first": True, "symptom_onset": "|%d %b %Y"},
    labels={
        "symptom_onset": "Fecha primer síntoma",
        "country": "País",
        "cases": "Casos totales",
        "days_after_first": "Días desde primer caso"
    }
)
fig_timeline.update_layout(
    plot_bgcolor="#0f0f1a",
    paper_bgcolor="#0a0a0f",
    font=dict(family="Space Mono", color="#e8e8f0", size=11),
    coloraxis_colorbar=dict(title="Días después<br>del primer caso", tickfont=dict(color="#e8e8f0")),
    xaxis=dict(showgrid=True, gridcolor="#1e1e3a", color="#e8e8f0", tickformat="%d %b %Y"),
    yaxis=dict(showgrid=False, color="#e8e8f0"),
    height=500,
    margin=dict(l=20, r=20, t=20, b=20),
)
fig_timeline.update_traces(marker=dict(line=dict(color="#0a0a0f", width=1)), opacity=0.9)
st.plotly_chart(fig_timeline, use_container_width=True)

# ─── CHART 2: Cases over time by country (area chart) ─────────────────────────
st.markdown("### 📈 Evolución de casos por país en el tiempo")

ts = (
    fdf.groupby(["country", pd.Grouper(key="symptom_onset", freq=freq)])
    .size()
    .reset_index(name="cases")
)

fig_area = px.area(
    ts,
    x="symptom_onset",
    y="cases",
    color="country",
    line_group="country",
    color_discrete_sequence=px.colors.qualitative.Dark24,
    labels={"symptom_onset": "Fecha", "cases": "Nuevos casos", "country": "País"},
)
fig_area.update_layout(
    plot_bgcolor="#0f0f1a",
    paper_bgcolor="#0a0a0f",
    font=dict(family="Space Mono", color="#e8e8f0", size=11),
    xaxis=dict(showgrid=True, gridcolor="#1e1e3a", color="#e8e8f0", tickformat="%d %b %Y"),
    yaxis=dict(showgrid=True, gridcolor="#1e1e3a", color="#e8e8f0"),
    legend=dict(bgcolor="#0f0f1a", bordercolor="#ff6b35", borderwidth=1, font=dict(color="#e8e8f0")),
    height=450,
    margin=dict(l=20, r=20, t=20, b=20),
    hovermode="x unified"
)
st.plotly_chart(fig_area, use_container_width=True)

# ─── CHART 3: Heatmap country × week ─────────────────────────────────────────
st.markdown("### 🔥 Mapa de calor — Intensidad por semana y país")

heat_ts = (
    fdf.groupby(["country", pd.Grouper(key="symptom_onset", freq="W")])
    .size()
    .reset_index(name="cases")
)
heat_pivot = heat_ts.pivot(index="country", columns="symptom_onset", values="cases").fillna(0)
heat_pivot.columns = heat_pivot.columns.strftime("%d %b")

fig_heat = go.Figure(go.Heatmap(
    z=heat_pivot.values,
    x=heat_pivot.columns.tolist(),
    y=heat_pivot.index.tolist(),
    colorscale=[[0, "#0f0f1a"], [0.3, "#3d1a0a"], [0.6, "#ff6b35"], [1, "#fff"]],
    hovertemplate="<b>%{y}</b><br>Semana: %{x}<br>Casos: %{z}<extra></extra>",
))
fig_heat.update_layout(
    plot_bgcolor="#0f0f1a",
    paper_bgcolor="#0a0a0f",
    font=dict(family="Space Mono", color="#e8e8f0", size=10),
    xaxis=dict(showgrid=False, color="#e8e8f0", tickangle=-45),
    yaxis=dict(showgrid=False, color="#e8e8f0"),
    height=max(300, len(heat_pivot) * 30),
    margin=dict(l=20, r=20, t=20, b=80),
)
st.plotly_chart(fig_heat, use_container_width=True)

# ─── CHART 4: Cumulative race ──────────────────────────────────────────────────
st.markdown("### 🏁 Acumulado de casos por país")

cumulative = (
    ts.sort_values("symptom_onset")
    .assign(cumulative=lambda d: d.groupby("country")["cases"].cumsum())
)
fig_cum = px.line(
    cumulative,
    x="symptom_onset",
    y="cumulative",
    color="country",
    color_discrete_sequence=px.colors.qualitative.Dark24,
    labels={"symptom_onset": "Fecha", "cumulative": "Casos acumulados", "country": "País"},
    log_y=True,
)
fig_cum.update_traces(line=dict(width=2))
fig_cum.update_layout(
    plot_bgcolor="#0f0f1a",
    paper_bgcolor="#0a0a0f",
    font=dict(family="Space Mono", color="#e8e8f0", size=11),
    xaxis=dict(showgrid=True, gridcolor="#1e1e3a", color="#e8e8f0"),
    yaxis=dict(showgrid=True, gridcolor="#1e1e3a", color="#e8e8f0", title="Casos (escala log)"),
    legend=dict(bgcolor="#0f0f1a", bordercolor="#ff6b35", borderwidth=1, font=dict(color="#e8e8f0")),
    height=400,
    margin=dict(l=20, r=20, t=20, b=20),
    hovermode="x unified"
)
st.plotly_chart(fig_cum, use_container_width=True)

# ─── TABLE: Ranking primeros casos ────────────────────────────────────────────
st.markdown("### 📋 Ranking — Orden de aparición por país")
ranking = timeline_df[["country", "symptom_onset", "days_after_first", "cases"]].copy()
ranking.columns = ["País", "Primer síntoma", "Días desde China", "Total casos"]
ranking["Primer síntoma"] = ranking["Primer síntoma"].dt.strftime("%d %b %Y")
ranking = ranking.reset_index(drop=True)
ranking.index += 1

st.dataframe(
    ranking,
    use_container_width=True,
    height=350,
    column_config={
        "Días desde China": st.column_config.ProgressColumn(
            min_value=0,
            max_value=int(ranking["Días desde China"].max()),
            format="%d días",
        ),
        "Total casos": st.column_config.NumberColumn(format="%d"),
    }
)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#555; font-size:0.75rem; font-family:Space Mono'>COVID-19 Early Case Dashboard · Sube tu propio CSV con las columnas correctas para usar datos reales</div>",
    unsafe_allow_html=True
)
