import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings

# -----------------------------------------------------------------------------
# 1. DEPLOYMENT CONFIGURATION (Must be first Streamlit command)
# -----------------------------------------------------------------------------
warnings.filterwarnings('ignore')
st.set_page_config(page_title="USDA AI Insights Suite", layout="wide")

# -----------------------------------------------------------------------------
# 2. REQUIRED FUNCTIONS (Architectural Logic)
# -----------------------------------------------------------------------------

@st.cache_data
def load_and_filter_data(filepath="usda_data_clean.csv"):
    """RULE 1: STRICT VOLUME THRESHOLDING (Applied at Source)"""
    try:
        df = pd.read_csv(filepath)
        # Note: Data is pre-cleaned to exclude sessions < 50k.
        # We retain all rows here as the file represents the systemic population.
        return df
    except FileNotFoundError:
        st.error(f"Critical Error: '{filepath}' not found in root directory.")
        st.stop()
    except Exception as e:
        st.error(f"Critical Error loading data: {e}")
        st.stop()

@st.cache_data
def create_diagnostics(df_rd, features):
    """Generate static Elbow and Silhouette plots for technical validation"""
    X = df_rd[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    silhouettes = []
    K_range = range(2, 7)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
        
    fig_elbow = px.line(x=list(K_range), y=inertias, markers=True, 
                        title="Elbow Method (Inertia)", labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'})
    
    fig_sil = px.line(x=list(K_range), y=silhouettes, markers=True, 
                      title="Silhouette Score Validation", labels={'x': 'Number of Clusters (k)', 'y': 'Silhouette Score'})
    fig_sil.update_traces(line_color='green')
    
    return fig_elbow, fig_sil

@st.cache_data
def preprocess_and_cluster(df_rd, features):
    """RULE 3 & MODELING RULES: Standardize, Cluster, and Map Personas"""
    # Create copy to avoid SettingWithCopy warnings
    df_model = df_rd.copy()
    X = df_model[features]
    
    # Standardize BEFORE KMeans
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Force n_clusters = 3 per project requirements
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_model['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Persona Logic: Map by Bounce Rate (Well-Served = Low, Underserved = High)
    cluster_means = df_model.groupby('Cluster')[features].mean()
    sorted_indices = cluster_means.sort_values(by='Total Bounce rate').index
    
    persona_mapping = {
        sorted_indices[0]: "Well-Served",
        sorted_indices[1]: "Moderately Served",
        sorted_indices[2]: "Underserved"
    }
    
    df_model['Persona'] = df_model['Cluster'].map(persona_mapping)
    
    # Prepare scaled centroids for Radar Charts
    centroids_scaled = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    centroids_scaled['Persona'] = centroids_scaled.index.map(persona_mapping)
    
    return df_model, scaler, kmeans, persona_mapping, centroids_scaled

def plot_friction_matrix(df_model):
    """Plotly Scatter Plot of Duration vs Bounce Rate with Zombie Session Annotation"""
    fig = px.scatter(
        df_model, 
        x='Total Average session duration', 
        y='Total Bounce rate', 
        color='Persona',
        hover_data=['Page title'],
        title="Friction Matrix: Session Duration vs Bounce Rate",
        labels={'Total Average session duration': 'Avg Session Duration (s)', 'Total Bounce rate': 'Bounce Rate'},
        color_discrete_map={"Well-Served": "green", "Moderately Served": "orange", "Underserved": "red"}
    )
    
    # Highlight "Zombie Sessions" (Midpoint Insight)
    fig.add_vrect(
        x0=1000, x1=df_model['Total Average session duration'].max() + 100,
        fillcolor="red", opacity=0.1, line_width=0,
        annotation_text="Zombie Sessions (>1000s)", annotation_position="top right"
    )
    
    fig.update_layout(height=600)
    return fig

def run_simulator(scaler, kmeans, persona_mapping, features):
    """Tab 3 Logic: Interactive Strategic AI Roadmap Simulator"""
    st.subheader("Simulate Hypothetical Page Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sim_duration = st.slider("Average Session Duration (s)", 0, 2000, 150)
    with col2:
        sim_bounce = st.slider("Bounce Rate", 0.0, 1.0, 0.40)
    with col3:
        sim_views = st.slider("Views per Session", 1.0, 10.0, 2.5)
        
    # Input data array must match scaler feature order
    input_data = pd.DataFrame([[sim_views, sim_bounce, sim_duration]], columns=features)
    
    if st.button("Predict Persona & Generate AI Roadmap", type="primary"):
        input_scaled = scaler.transform(input_data)
        pred_cluster = kmeans.predict(input_scaled)[0]
        assigned_persona = persona_mapping[pred_cluster]
        
        st.divider()
        st.markdown(f"### 🎯 Predicted Persona: **{assigned_persona}**")
        
        if assigned_persona == "Underserved":
            st.error("**Prioritization: HIGH**")
            st.warning("**AI Roadmap Action:** Deploy AI-Enabled Guided Navigation and Proactive Chatbots to immediately intercept high-friction drop-offs.")
        elif assigned_persona == "Moderately Served":
            st.warning("**Prioritization: MEDIUM**")
            st.info("**AI Roadmap Action:** Implement AI Search Optimization and Dynamic Content Recommendations to clarify ambiguous navigation pathways.")
        else:
            st.success("**Prioritization: LOW**")
            st.success("**AI Roadmap Action:** Maintain current architecture. Apply periodic AI-driven behavior monitoring to ensure engagement stability.")

# -----------------------------------------------------------------------------
# 3. MAIN APP EXECUTION
# -----------------------------------------------------------------------------

st.title("USDA Digital Pathway & AI Insights Suite")

# RULE 2: DOMESTIC GEOGRAPHY ANCHOR
st.info("📊 **Global Context Rule:** 99.57% Domestic (US) Traffic Consistency. Analysis focuses exclusively on domestic behavioral personas.")

# Load Pre-Cleaned Data
df = load_and_filter_data("usda_data_clean.csv")

# RULE 3: SAFE SUBSETTING FOR CLUSTERING
df_rd_raw = df[df['Is_RD'] == True].copy()
clustering_features = ['Total Views per session', 'Total Bounce rate', 'Total Average session duration']

# Data Cleaning for Model
df_rd_raw = df_rd_raw.dropna(subset=clustering_features)

if df_rd_raw.empty:
    st.error("⚠️ **Data Error:** No Rural Development pages found. Please verify the 'Is_RD' flag in your source dataset.")
    st.stop()

# Run Analytical Engine
df_model, fitted_scaler, fitted_kmeans, p_mapping, scaled_centroids = preprocess_and_cluster(df_rd_raw, clustering_features)

# APP STRUCTURE (STRICT)
tab1, tab2, tab3 = st.tabs(["System-Wide Briefing", "RD Behavioral Clustering", "Strategic AI Simulator"])

with tab1:
    st.header("Executive Summary: System-Wide Briefing")
    c1, c2 = st.columns(2)
    with c1:
        trend_df = df.groupby(['Month', 'Day'])['Total Sessions'].sum().reset_index()
        trend_df['Timeline'] = trend_df['Month'].astype(str) + "/" + trend_df['Day'].astype(str)
        st.plotly_chart(px.line(trend_df, x='Timeline', y='Total Sessions', title="Traffic Trends (Filtered System-Wide)"), use_container_width=True)
    with c2:
        device_df = df[['Page title', 'Desktop Bounce rate', 'Mobile Bounce rate']].head(15).melt(id_vars='Page title')
        fig_dev = px.bar(device_df, x='Page title', y='value', color='variable', barmode='group', title="Device Friction: Mobile vs. Desktop")
        fig_dev.add_annotation(text="The Vital 3-Minute Window (0-200s)", xref="paper", yref="paper", x=0.5, y=-0.3, showarrow=False)
        st.plotly_chart(fig_dev, use_container_width=True)

with tab2:
    st.header("RD Behavioral Clustering & Diagnostics")
    
    st.subheader("Model Validation")
    diag_c1, diag_c2 = st.columns(2)
    f_elb, f_sil = create_diagnostics(df_rd_raw, clustering_features)
    diag_c1.plotly_chart(f_elb, use_container_width=True)
    diag_c2.plotly_chart(f_sil, use_container_width=True)
    
    st.divider()
    st.subheader("Persona Profiling")
    fig_radar = go.Figure()
    for _, row in scaled_centroids.iterrows():
        fig_radar.add_trace(go.Scatterpolar(r=[row[feat] for feat in clustering_features], theta=clustering_features, fill='toself', name=row['Persona']))
    st.plotly_chart(fig_radar, use_container_width=True)
    
    st.divider()
    st.subheader("Friction Matrix Mapping")
    st.plotly_chart(plot_friction_matrix(df_model), use_container_width=True)

with tab3:
    st.header("Strategic AI Simulator")
    run_simulator(fitted_scaler, fitted_kmeans, p_mapping, clustering_features)
