import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings

# -----------------------------------------------------------------------------
# 1. DEPLOYMENT CONFIGURATION
# -----------------------------------------------------------------------------
warnings.filterwarnings('ignore')
st.set_page_config(page_title="USDA AI Insights Suite", layout="wide")

# -----------------------------------------------------------------------------
# 2. CORE ARCHITECTURAL FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_data
def load_and_filter_data(filepath="usda_data_clean.csv", threshold=0):
    """
    Source of Truth: Loads the pre-cleaned dataset.
    The threshold is parameterized to ensure data integrity during clustering.
    """
    try:
        df = pd.read_csv(filepath)
        # Apply threshold only if specific volume isolation is required
        df_filtered = df[df['Total Sessions'] >= threshold].copy()
        return df_filtered
    except FileNotFoundError:
        st.error("Critical Error: 'usda_data_clean.csv' not found.")
        st.stop()
    except Exception as e:
        st.error(f"Critical Error: {e}")
        st.stop()

@st.cache_data
def run_analytics_engine(df_rd, features):
    """
    Standardization & Clustering Pipeline.
    Strict mapping to Well-Served, Moderately Served, and Underserved personas.
    """
    X = df_rd[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means logic fixed to n=3 for persona mapping
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_rd['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Logic: Lower Bounce Rate/Higher Duration = Well-Served
    cluster_means = df_rd.groupby('Cluster')[features].mean()
    sorted_indices = cluster_means.sort_values(by='Total Bounce rate').index
    
    persona_mapping = {
        sorted_indices[0]: "Well-Served",
        sorted_clusters[1] if len(sorted_indices) > 1 else sorted_indices[0]: "Moderately Served",
        sorted_indices[-1]: "Underserved"
    }
    
    df_rd['Persona'] = df_rd['Cluster'].map(persona_mapping)
    
    # Prepare scaled centroids for Radar Charts
    centroids_scaled = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    centroids_scaled['Persona'] = centroids_scaled.index.map(persona_mapping)
    
    return df_rd, scaler, kmeans, persona_mapping, centroids_scaled

# -----------------------------------------------------------------------------
# 3. UI ASSEMBLY & EXECUTION
# -----------------------------------------------------------------------------

st.title("USDA Digital Pathway & AI Insights Suite")

# DATA FACT SHEET - Hardcoded per midpoint findings
st.info("📊 **Analysis Parameter:** Data is pre-filtered for high-impact systemic traffic. 99.57% Domestic (US) Traffic Consistency confirmed.")

# Load pre-cleaned data
# Set threshold to 0 because the file is already pre-filtered for the desired volume
df = load_and_filter_data("usda_data_clean.csv", threshold=0)

# Isolate Rural Development (RD)
df_rd = df[df['Is_RD'] == True].copy()
clustering_features = ['Total Views per session', 'Total Bounce rate', 'Total Average session duration']

# Drop NaNs to prevent model failure
df_rd = df_rd.dropna(subset=clustering_features)

if df_rd.empty:
    st.error("⚠️ **Data Integrity Warning:** No Rural Development pages detected in the provided dataset. Check the 'Is_RD' flag in the source file.")
    st.stop()

# Run Engine
df_rd_clustered, fitted_scaler, fitted_kmeans, p_mapping, scaled_centroids = run_analytics_engine(df_rd, clustering_features)

tab1, tab2, tab3 = st.tabs(["System-Wide Briefing", "RD Behavioral Clustering", "Strategic AI Simulator"])

with tab1:
    st.header("System-Wide Executive Briefing")
    col1, col2 = st.columns(2)
    
    with col1:
        trend_df = df.groupby(['Month', 'Day'])['Total Sessions'].sum().reset_index()
        trend_df['Timeline'] = trend_df['Month'].astype(str) + "/" + trend_df['Day'].astype(str)
        st.plotly_chart(px.line(trend_df, x='Timeline', y='Total Sessions', title="Systemic Traffic Trends (2025)"), use_container_width=True)
        
    with col2:
        device_comp = df[['Page title', 'Desktop Bounce rate', 'Mobile Bounce rate']].head(15).melt(id_vars='Page title')
        st.plotly_chart(px.bar(device_comp, x='Page title', y='value', color='variable', barmode='group', title="Device Engagement Friction"), use_container_width=True)

with tab2:
    st.header("Rural Development Behavioral Profiling")
    st.subheader("Friction Matrix & User Personas")
    
    # Friction Matrix Plot
    fig_friction = px.scatter(df_rd_clustered, x='Total Average session duration', y='Total Bounce rate', color='Persona', hover_data=['Page title'])
    fig_friction.add_vrect(x0=1000, x1=df_rd_clustered['Total Average session duration'].max(), fillcolor="red", opacity=0.1, annotation_text="Zombie Sessions")
    st.plotly_chart(fig_friction, use_container_width=True)
    
    # Radar Charts
    fig_radar = go.Figure()
    for _, row in scaled_centroids.iterrows():
        fig_radar.add_trace(go.Scatterpolar(r=[row[f] for f in clustering_features], theta=clustering_features, fill='toself', name=row['Persona']))
    st.plotly_chart(fig_radar, use_container_width=True)

with tab3:
    st.header("Strategic AI Simulator")
    st.write("Input real-time page metrics to generate data-driven AI Roadmap recommendations.")
    
    c1, c2, c3 = st.columns(3)
    in_duration = c1.slider("Avg Duration (s)", 0, 2000, 150)
    in_bounce = c2.slider("Bounce Rate", 0.0, 1.0, 0.4)
    in_views = c3.slider("Views / Session", 1.0, 10.0, 2.0)
    
    input_scaled = fitted_scaler.transform([[in_views, in_bounce, in_duration]])
    prediction = p_mapping[fitted_kmeans.predict(input_scaled)[0]]
    
    st.divider()
    st.subheader(f"Assigned Persona: {prediction}")
    
    if prediction == "Underserved":
        st.error("**Prioritization: HIGH** | Deploy AI-Enabled Guided Navigation and Proactive Chatbots.")
    elif prediction == "Moderately Served":
        st.warning("**Prioritization: MEDIUM** | Implement AI Search Optimization and Dynamic Content Recommendations.")
    else:
        st.success("**Prioritization: LOW** | Maintain architecture; apply AI-driven behavior monitoring.")
