import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data_loader import load_data, get_stats
from src.model import detect_anomalies

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DataGuard Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Light Mode / Clean Professional Look)
st.markdown("""
<style>
    /* Global tweaks - Light Background */
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
    }
    
    /* Metrics 'Cards' */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Headers alignment */
    h1, h2, h3 {
        color: #0d6efd; /* Bootstrap Primary Blue */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #dee2e6;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #ffffff;
        border-radius: 4px;
        border: 1px solid #dee2e6;
        padding: 10px 20px;
        color: #495057;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0d6efd !important;
        color: white !important;
    }
    
    /* DataFrame Highlight */
    .dataframe {
        font-family: sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Data Loading & Sidebar Configuration
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_prep_data():
    return load_data("ventes.csv")

df = load_and_prep_data()

if df.empty:
    st.error("❌ Impossible de charger les données 'ventes.csv'.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/data-shield.png", width=60)
    st.title("DataGuard Pro")
    st.markdown("---")
    
    # 1. Store Selector
    stores = sorted(df['magasin'].unique())
    selected_store = st.selectbox("📍 Sélectionner un magasin", stores)
    
    # 2. Date Range Filter
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    st.subheader("📅 Période d'analyse")
    date_range = st.date_input(
        "Filtrer par date",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # 3. Model Sensitivity
    st.subheader("⚙️ Paramètres IA")
    contamination = st.slider(
        "Sensibilité (Contamination)", 
        min_value=0.01, 
        max_value=0.20, 
        value=0.05,
        step=0.01
    )
    
    st.markdown("---")
    st.caption(f"Données chargées : {len(df)} lignes")

# -----------------------------------------------------------------------------
# 3. Filtering & Logic
# -----------------------------------------------------------------------------
# Filter by Store
df_store = df[df['magasin'] == selected_store].copy()

# Filter by Date
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df_store['date'].dt.date >= start_date) & (df_store['date'].dt.date <= end_date)
    df_store = df_store.loc[mask]

df_store = df_store.sort_values('date')

# Run Model
if len(df_store) > 10:  # Need enough data for IsolationForest
    df_store = detect_anomalies(df_store, contamination=contamination)
else:
    st.warning("⚠️ Pas assez de données sur cette période pour lancer l'IA.")
    df_store['anomaly'] = False
    df_store['anomaly_score'] = 1

# Calculate Stats
stats = get_stats(df_store)
anomalies_count = df_store['anomaly'].sum()
anomalies_df = df_store[df_store['anomaly']].copy()

# -----------------------------------------------------------------------------
# 4. Main Dashboard Layout
# -----------------------------------------------------------------------------

st.title(f"📊 Analyse : {selected_store}")

# --- KPIs ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Ventes Moyennes", f"{stats.get('mean', 0):.2f} €")
with kpi2:
    st.metric("Ventes Max", f"{stats.get('max', 0):.2f} €")
with kpi3:
    st.metric("Ventes Min", f"{stats.get('min', 0):.2f} €")
with kpi4:
    st.metric("Anomalies Détectées", f"{anomalies_count}", delta_color="inverse")

st.markdown("###") # Spacer

# --- Tabs for different views ---
tab1, tab2, tab3 = st.tabs(["📈 Vue Temporelle", "📊 Analyse Détaillée", "📋 Données Brutes"])

# TAB 1: Main Time Series
with tab1:
    st.subheader("Évolution dans le temps")
    st.caption("Les croix rouges indiquent les valeurs détectées comme anormales.")
    
    fig = go.Figure()

    # Line for normal sales
    fig.add_trace(go.Scatter(
        x=df_store['date'], 
        y=df_store['ventes'],
        mode='lines',
        name='Ventes',
        line=dict(color='#0d6efd', width=2)
    ))

    # Anomalies markers
    if not anomalies_df.empty:
        fig.add_trace(go.Scatter(
            x=anomalies_df['date'],
            y=anomalies_df['ventes'],
            mode='markers',
            name='Anomalie',
            marker=dict(color='#dc3545', size=10, symbol='x', line=dict(width=2, color='white'))
        ))

    fig.update_layout(
        template="plotly_white", # Light theme
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Date",
        yaxis_title="Montant des Ventes (€)",
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center'),
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e9ecef')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e9ecef')
    
    st.plotly_chart(fig, use_container_width=True)

# TAB 2: Advanced Analysis (Histogram + Weekly)
with tab2:
    col_a, col_b = st.columns(2)
    
    # Histogram of Sales
    with col_a:
        st.subheader("Distribution des Montants")
        fig_hist = px.histogram(
            df_store, 
            x="ventes", 
            nbins=30, 
            color="anomaly",
            color_discrete_map={False: "#0d6efd", True: "#dc3545"},
            title="Histogramme des Ventes",
            template="plotly_white"
        )
        fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_hist, use_container_width=True)
        
    # Day of Week Analysis
    with col_b:
        st.subheader("Anomalies par Jour")
        df_store['day_of_week'] = df_store['date'].dt.day_name()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Count anomalies per day
        anom_per_day = df_store[df_store['anomaly']].groupby('day_of_week').size().reindex(days_order, fill_value=0).reset_index(name='count')
        
        fig_bar = px.bar(
            anom_per_day, 
            x='day_of_week', 
            y='count',
            title="Fréquence par Jour de Semaine",
            template="plotly_white",
            color_discrete_sequence=['#ffc107'] # Warning color
        )
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)

# TAB 3: Data Table
with tab3:
    st.subheader("Détail des Anomalies")
    
    if not anomalies_df.empty:
        st.dataframe(
            anomalies_df[['date', 'magasin', 'ventes', 'anomaly_score']].style
            .format({'ventes': "{:.2f} €", 'anomaly_score': "{:.4f}"})
            .background_gradient(subset=['ventes'], cmap='Reds'),
            use_container_width=True
        )
    else:
        st.success("✅ Aucune anomalie détectée sur cette période.")
