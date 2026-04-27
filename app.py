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
    """RULE 1: VOLUME THRESHOLDING"""
    try:
        df = pd.read_csv(filepath)
        # NOTE: If the app throws the empty data error, lower this 50000 number!
        # For deployment testing, you might need to drop it to 5000 or 1000 depending on your clean data.
        df_filtered = df[df['Total Sessions'] >= 50000].copy()
        return df_filtered
    except FileNotFoundError:
        st.error("Critical Error: 'usda_data_clean.csv' not found. Please ensure the dataset is in the same directory as app.py.")
        st.stop()
    except Exception as e:
        st.error(f"Critical Error loading data: {e}")
        st.stop()

@st.cache_data
def create_diagnostics(df_rd, features):
    """Generate the Elbow Method and Silhouette plots for k=2 to k=6 (STATIC)"""
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
    X = df_rd[features]
    
    # Standardize BEFORE KMeans
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Force n_clusters = 3
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_rd['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Calculate means to map exactly to the required Personas
    cluster_means = df_rd.groupby('Cluster')[features].mean()
    
    # Logic to map: Highest Bounce Rate = Underserved, Lowest = Well-Served
    sorted_clusters = cluster_means.sort_values(by='Total Bounce rate').index
    
    persona_mapping = {
        sorted_clusters[0]: "Well-Served",
        sorted_clusters[1]: "Moderately Served",
        sorted_clusters[2]: "Underserved"
    }
    
    df_rd['Persona'] = df_rd['Cluster'].map(persona_mapping)
    
    # Get standard scaled means for the Radar Chart to ensure equal visual weighting
    centroids_scaled = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    centroids_scaled['Persona'] = centroids_scaled.index.map(persona_mapping)
    
    return df_rd, scaler, kmeans, persona_mapping, centroids_scaled

def plot_friction_matrix(df_rd):
    """Plotly Scatter Plot of Duration vs Bounce Rate with Zombie Session Annotation"""
    fig = px.scatter(
        df_rd, 
        x='Total Average session duration', 
        y='Total Bounce rate', 
        color='Persona',
        hover_data=['Page title'],
        title="Friction Matrix: Session Duration vs Bounce Rate",
        labels={'Total Average session duration': 'Avg Session Duration (s)', 'Total Bounce rate': 'Bounce Rate'}
    )
    
    # Highlight "Zombie Sessions"
    fig.add_vrect(
        x0=1000, x1=df_rd['Total Average session duration'].max() + 100,
        fillcolor="red", opacity=0.1, line_width=0,
        annotation_text="Zombie Sessions (>1000s)", annotation_position="top right"
    )
    
    # Ensure UI requirement of large readable charts
    fig.update_layout(height=600)
    return fig

def run_simulator(scaler, kmeans, persona_mapping, features):
    """Strategic AI Simulator UI and Logic"""
    st.subheader("Simulate Hypothetical Page Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sim_duration = st.slider("Average Session Duration (s)", 0, 2000, 150)
    with col2:
        sim_bounce = st.slider("Bounce Rate", 0.0, 1.0, 0.40)
    with col3:
        sim_views = st.slider("Views per Session", 1.0, 10.0, 2.5)
        
    # MUST match 'features' exact ordering
    input_data = pd.DataFrame([[sim_views, sim_bounce, sim_duration]], columns=features)
    
    if st.button("Predict Persona & Generate AI Roadmap", type="primary"):
        input_scaled = scaler.transform(input_data)
        pred_cluster = kmeans.predict(input_scaled)[0]
        assigned_persona = persona_mapping[pred_cluster]
        
        st.markdown("---")
        st.markdown(f"### 🎯 Predicted Persona: **{assigned_persona}**")
        
        if assigned_persona == "Underserved":
            st.error("**Prioritization:** HIGH")
            st.warning("**AI Roadmap Action:** Implement AI-Enabled Guided Navigation and Proactive Chatbots to immediately intercept drop-offs and reduce acute friction.")
        elif assigned_persona == "Moderately Served":
            st.warning("**Prioritization:** MEDIUM")
            st.info("**AI Roadmap Action:** Deploy AI Search Optimization and Dynamic Content Recommendations to clarify ambiguous navigation pathways.")
        else: # Well-Served
            st.success("**Prioritization:** LOW")
            st.success("**AI Roadmap Action:** Maintain current architecture. Implement periodic AI-driven behavior monitoring to ensure engagement remains stable.")

# -----------------------------------------------------------------------------
# 3. MAIN APP EXECUTION (UI Assembly)
# -----------------------------------------------------------------------------

st.title("USDA Digital Pathway & AI Insights Suite")

st.info("📊 **Global Context Rule:** 99.57% Domestic (US) Traffic Consistency. International traffic is statistically negligible; the following models map exclusively domestic behavior patterns.")

# Load Data
df = load_and_filter_data("usda_data_clean.csv")

# RULE 3: SAFE SUBSETTING FOR CLUSTERING
df_rd = df[df['Is_RD'] == True].copy()
clustering_features = ['Total Views per session', 'Total Bounce rate', 'Total Average session duration']

# --- DEPLOYMENT SAFETY CATCH (Fixes the ValueError) ---
df_rd = df_rd.dropna(subset=clustering_features)

if df_rd.empty:
    st.error("⚠️ **Data Filter Error:** There are no Rural Development pages that meet the `Total Sessions` threshold. Please open your `app.py` code, locate `load_and_filter_data()`, and lower the `50000` threshold to a smaller number (like `5000` or `1000`) so the clustering algorithm has data to work with.")
    st.stop()
# ------------------------------------------------------

# Preprocess and prepare models
df_rd_clustered, fitted_scaler, fitted_kmeans, p_mapping, scaled_centroids = preprocess_and_cluster(df_rd, clustering_features)

# APP STRUCTURE (STRICT)
tab1, tab2, tab3 = st.tabs(["System-Wide Briefing", "RD Behavioral Clustering", "Strategic AI Simulator"])

with tab1:
    st.header("System-Wide Briefing")
    
    colA, colB = st.columns(2)
    with colA:
        trend_df = df.groupby(['Month', 'Day'])['Total Sessions'].sum().reset_index()
        trend_df['Date_Proxy'] = trend_df['Month'].astype(str) + "/" + trend_df['Day'].astype(str)
        fig_trends = px.line(trend_df, x='Date_Proxy', y='Total Sessions', title="Traffic Trends (Filtered System-Wide)")
        st.plotly_chart(fig_trends, use_container_width=True)
        
    with colB:
        device_df = df[['Page title', 'Desktop Bounce rate', 'Mobile Bounce rate']].head(20).melt(
            id_vars=['Page title'], var_name='Device', value_name='Bounce Rate'
        )
        fig_device = px.bar(device_df, x='Page title', y='Bounce Rate', color='Device', barmode='group', 
                            title="Device Friction: Mobile vs. Desktop Bounce Rates (Top Pages)")
        fig_device.add_annotation(text="Critical focus area: The Vital 3-Minute Window (0-200s)", 
                                  xref="paper", yref="paper", x=0.5, y=-0.3, showarrow=False)
        st.plotly_chart(fig_device, use_container_width=True)

with tab2:
    st.header("RD Behavioral Clustering")
    
    st.subheader("Model Diagnostics (Static Validation)")
    col1, col2 = st.columns(2)
    fig_elb, fig_sil = create_diagnostics(df_rd, clustering_features)
    with col1:
        st.plotly_chart(fig_elb, use_container_width=True)
    with col2:
        st.plotly_chart(fig_sil, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Persona Profiling (Radar Charts)")
    fig_radar = go.Figure()
    for index, row in scaled_centroids.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row[feat] for feat in clustering_features],
            theta=clustering_features,
            fill='toself',
            name=row['Persona']
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Standardized Feature Means by Persona")
    st.plotly_chart(fig_radar, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Friction Mapping")
    fig_matrix = plot_friction_matrix(df_rd_clustered)
    st.plotly_chart(fig_matrix, use_container_width=True)

with tab3:
    st.header("Strategic AI Simulator")
    st.markdown("Adjust the page metrics below to classify the user behavior and receive automated AI roadmap priorities.")
    run_simulator(fitted_scaler, fitted_kmeans, p_mapping, clustering_features)
