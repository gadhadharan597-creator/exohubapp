import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Exoplanet Habitability Explorer",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme styling
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .stMetric {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    .stMetric label {
        color: #94a3b8 !important;
        font-weight: 600;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 700;
    }
    h1, h2, h3 {
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    .highlight-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #6366f1;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title Header
st.title("🪐 Advanced Exoplanet Habitability Ranking System")
st.markdown("""
An interactive multi-stage ranking framework combining **NASA Exoplanet Archive** parameters with **NASA ExoMiner++** planet validation probabilities and **Kopparapu et al. Habitable Zone models**
""")
st.markdown("---")

# Load Data
DATA_FILE = "exoplanet_habitability_rankings.csv"

@st.cache_data
def load_rankings():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        return df
    else:
        st.error(f"Data file '{DATA_FILE}' not found. Please run 'exoplanet_pipeline.py' first.")
        return pd.DataFrame()

def create_orbit_visualization(planet_data, system_name):
    """Create a 3D orbit visualization for a multi-planet system"""
    
    # Normalize orbital period for visualization (AU-based)
    semi_major_axis = planet_data['pl_orbper'].values ** (2/3)  # Kepler's 3rd law approximation
    
    # Create 3D orbit
    theta = np.linspace(0, 2*np.pi, 1000)
    
    fig = go.Figure()
    
    # Add star at center
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=20, color='gold', symbol='circle'),
        name='Host Star',
        hovertemplate="<b>Host Star</b><br>Teff: ~5778K<extra></extra>"
    ))
    
    # Add planets and their orbits
    colors = ['#38bdf8', '#f43f5e', '#22c55e', '#a855f7', '#f59e0b', '#06b6d4']
    
    for idx, (_, planet) in enumerate(planet_data.iterrows()):
        a = semi_major_axis[idx]
        e = 0.1  # Assume small eccentricity for visualization
        
        # Elliptical orbit
        r = a * (1 - e**2) / (1 + e * np.cos(theta))
        x_orbit = r * np.cos(theta)
        y_orbit = r * np.sin(theta)
        z_orbit = np.zeros_like(theta)
        
        # Add orbit line
        fig.add_trace(go.Scatter3d(
            x=x_orbit, y=y_orbit, z=z_orbit,
            mode='lines',
            name=f'{planet["pl_name"]} Orbit',
            line=dict(color=colors[idx % len(colors)], width=2, dash='dash'),
            hoverinfo='skip'
        ))
        
        # Add planet position
        planet_x = a * (1 - e**2) / (1 + e)
        radius_scaled = max(0.01, planet['pl_rade'] / 5)  # Scale for visibility
        
        fig.add_trace(go.Scatter3d(
            x=[planet_x], y=[0], z=[0],
            mode='markers+text',
            marker=dict(
                size=radius_scaled * 8,
                color=colors[idx % len(colors)],
                line=dict(color='white', width=1)
            ),
            text=[planet['pl_name']],
            textposition='top center',
            name=planet['pl_name'],
            hovertemplate=f"<b>{planet['pl_name']}</b><br>" +
                         f"Radius: {planet['pl_rade']:.2f} R⊕<br>" +
                         f"Insolation: {planet['pl_insol']:.2f} S⊕<br>" +
                         f"Composite Score: {planet['composite_habitability_score']:.3f}<extra></extra>"
        ))
    
    # Add habitable zone (shaded region)
    hz_inner = 0.5  # Approximate
    hz_outer = 1.5  # Approximate
    hz_theta = np.linspace(0, 2*np.pi, 100)
    
    for r in np.linspace(hz_inner, hz_outer, 5):
        x_hz = r * np.cos(hz_theta)
        y_hz = r * np.sin(hz_theta)
        z_hz = np.zeros_like(hz_theta)
        fig.add_trace(go.Scatter3d(
            x=x_hz, y=y_hz, z=z_hz,
            mode='lines',
            name='Habitable Zone' if r == hz_inner else '',
            line=dict(color='rgba(34, 197, 94, 0.2)', width=1, dash='dot'),
            showlegend=(r == hz_inner),
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=f"🌍 Multi-Planet System: {system_name}",
        scene=dict(
            xaxis_title="Distance (AU)",
            yaxis_title="Distance (AU)",
            zaxis_title="Distance (AU)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            bgcolor="rgba(11, 15, 25, 0.9)",
            xaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)", showgrid=True, zeroline=False),
            yaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)", showgrid=True, zeroline=False),
            zaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)", showgrid=True, zeroline=False),
        ),
        paper_bgcolor="rgba(11, 15, 25, 1)",
        font=dict(color="white", size=12),
        hovermode='closest',
        height=600,
        width=900,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    return fig

df = load_rankings()

if not df.empty:
    # Sidebar Filters
    st.sidebar.header("🔍 Filter Parameters")
    
    # Habitability Composite Score Slider
    min_composite = st.sidebar.slider(
        "Minimum Composite Habitability Score",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05
    )
    
    # Minimum ExoMiner Probability
    min_exominer = st.sidebar.slider(
        "Minimum Real Planet Probability (P_real)",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05
    )
    
    # Stellar Type Filter
    stellar_types = sorted(df['stellar_type'].dropna().unique().tolist())
    selected_stellar = st.sidebar.multiselect(
        "Select Host Star Spectral Types",
        options=stellar_types,
        default=stellar_types
    )
    
    # Boolean Checkboxes
    rocky_only = st.sidebar.checkbox("Show Rocky Planets Only (R ≤ 1.6 R⊕)", value=False)
    hz_only = st.sidebar.checkbox("Show Conservative Habitable Zone Planets Only", value=False)
    
    # Apply Filters
    filtered_df = df[
        (df['composite_habitability_score'] >= min_composite) &
        (df['P_real_planet'] >= min_exominer) &
        (df['stellar_type'].isin(selected_stellar))
    ].copy()
    
    if rocky_only:
        filtered_df = filtered_df[filtered_df['is_rocky'] == 1]
    
    if hz_only:
        filtered_df = filtered_df[filtered_df['P_HZ'] == 1]
    
    # Display Metrics Overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Catalog Candidates", f"{len(df):,}")
    with col2:
        st.metric("Filtered Candidates", f"{len(filtered_df):,}")
    with col3:
        high_cand = (df['composite_habitability_score'] >= 0.85).sum()
        st.metric("Top Tier Candidates (Score ≥ 0.85)", f"{high_cand}")
    with col4:
        rocky_count = (filtered_df['is_rocky'] == 1).sum()
        st.metric("Filtered Rocky Planets", f"{rocky_count}")
    
    st.markdown("### 🏆 Ranked Potentially Habitable Candidates")
    st.write(f"Showing **{len(filtered_df)}** exoplanet candidates meeting active filter criteria.")
    
    # Data Table Formatting
    display_cols = [
        'pl_name', 'hostname', 'stellar_type', 'pl_rade', 'eq_temp_k',
        'pl_insol', 'earth_similarity_index', 'P_real_planet', 'P_HZ',
        'physics_habitability_score', 'ml_habitability_score', 'composite_habitability_score'
    ]
    
    # Filter available display columns
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    
    renamed_cols = {
        'pl_name': 'Planet Name',
        'hostname': 'Host Star',
        'stellar_type': 'Star Type',
        'pl_rade': 'Radius (R⊕)',
        'eq_temp_k': 'T_eq (K)',
        'pl_insol': 'Insolation (S⊕)',
        'earth_similarity_index': 'Proxy ESI',
        'P_real_planet': 'P(Real Planet)',
        'P_HZ': 'In HZ',
        'physics_habitability_score': 'Physics Score',
        'ml_habitability_score': 'ML Score',
        'composite_habitability_score': 'Composite Score'
    }
    
    display_df = filtered_df[display_cols].rename(columns=renamed_cols)
    
    # Format numeric columns
    format_dict = {
        'Radius (R⊕)': '{:.2f}',
        'T_eq (K)': '{:.1f}',
        'Insolation (S⊕)': '{:.2f}',
        'Proxy ESI': '{:.3f}',
        'P(Real Planet)': '{:.3f}',
        'Physics Score': '{:.3f}',
        'ML Score': '{:.3f}',
        'Composite Score': '{:.3f}'
    }
    
    for col, fmt in format_dict.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: fmt.format(x) if pd.notnull(x) else 'N/A')
    
    st.dataframe(display_df, use_container_width=True)
    
    # Visualization Section
    st.markdown("---")
    st.subheader("📊 Habitability Distribution & Scientific Insights")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("#### Score Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.set_style("darkgrid")
        plt.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white', 'figure.facecolor': '#0b0f19', 'axes.facecolor': '#1e293b'})
        sns.histplot(filtered_df['composite_habitability_score'], kde=True, color='#38bdf8', ax=ax, bins=20)
        ax.set_title("Distribution of Composite Habitability Scores", color='white')
        ax.set_xlabel("Composite Habitability Score", color='white')
        ax.set_ylabel("Count", color='white')
        st.pyplot(fig)
    
    with chart_col2:
        st.markdown("#### Radius vs Insolation (Habitable Zone)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        plt.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white', 'figure.facecolor': '#0b0f19', 'axes.facecolor': '#1e293b'})
        scatter = ax2.scatter(
            filtered_df['pl_insol'],
            filtered_df['pl_rade'],
            c=filtered_df['composite_habitability_score'],
            cmap='viridis',
            alpha=0.8,
            edgecolors='w',
            linewidth=0.5
        )
        ax2.set_xscale('log')
        ax2.set_title("Planet Radius vs. Insolation Flux", color='white')
        ax2.set_xlabel("Insolation (Earth Flux Units, Log Scale)", color='white')
        ax2.set_ylabel("Radius (Earth Radii)", color='white')
        ax2.axhline(1.6, color='#f43f5e', linestyle='--', label='Rocky Threshold (1.6 R⊕)')
        ax2.legend()
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label("Composite Score", color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        st.pyplot(fig2)
    
    # Multi-Planet Orbit Visualizer
    st.markdown("---")
    st.subheader("🌍 Multi-Planet Orbit Visualizer")
    st.markdown("Explore 3D interactive orbit systems for multi-planet systems")
    
    # Get unique host stars with multiple planets
    star_planet_counts = filtered_df.groupby('hostname').size()
    multi_planet_systems = star_planet_counts[star_planet_counts > 1].index.tolist()
    
    if multi_planet_systems:
        selected_system = st.selectbox(
            "Select a Multi-Planet System to Visualize",
            options=multi_planet_systems,
            help="These systems have 2+ planets in our filtered dataset"
        )
        
        system_data = filtered_df[filtered_df['hostname'] == selected_system].copy()
        
        if not system_data.empty:
            # Create and display orbit visualization
            orbit_fig = create_orbit_visualization(system_data, selected_system)
            st.plotly_chart(orbit_fig, use_container_width=True)
            
            # Display system details
            st.markdown(f"#### System: {selected_system}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Planets", len(system_data))
            with col2:
                avg_score = system_data['composite_habitability_score'].mean()
                st.metric("Avg Habitability Score", f"{avg_score:.3f}")
            with col3:
                habitable_count = (system_data['P_HZ'] == 1).sum()
                st.metric("Planets in HZ", int(habitable_count))
    else:
        st.info("💡 No multi-planet systems found in filtered results. Adjust filters to see more systems.")

else:
    st.warning("No data loaded.")
