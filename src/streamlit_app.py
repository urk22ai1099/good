"""
AgroMind+ Streamlit Dashboard
Interactive web interface for farmers
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from integrated_system import AgroMindIntegratedSystem

# Page configuration
st.set_page_config(
    page_title="AgroMind+ | Smart Crop Advisory",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        font-weight: bold;
        padding: 1rem;
        background: linear-gradient(90deg, #81C784, #66BB6A);
        border-radius: 10px;
        color: white;
    }
    .crop-card {
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .rank-1 { border-color: #FFD700; background-color: #FFFACD; }
    .rank-2 { border-color: #C0C0C0; background-color: #F5F5F5; }
    .rank-3 { border-color: #CD7F32; background-color: #FFF8DC; }
    .rank-4 { border-color: #8BC34A; background-color: #F1F8E9; }
    
    .metric-card {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'system' not in st.session_state:
        st.session_state.system = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'selected_crop' not in st.session_state:
        st.session_state.selected_crop = None
    if 'advisory_report' not in st.session_state:
        st.session_state.advisory_report = None

def main():
    """Main dashboard function"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🌾 AgroMind+ | LSTM-Based Smart Crop Advisory</div>', 
                unsafe_allow_html=True)
    
    st.markdown("### AI-Powered Crop Recommendation & Advisory System")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/4CAF50/FFFFFF?text=AgroMind%2B", 
                use_column_width=True)
        
        st.markdown("## 📍 Farm Information")
        
        farm_id = st.text_input("Farm ID", value="FARM_001")
        farm_size = st.number_input("Farm Size (hectares)", min_value=0.1, 
                                    max_value=100.0, value=2.0, step=0.1)
        region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
        
        st.markdown("---")
        st.markdown("## 📊 Data Source")
        data_mode = st.radio("", ["Manual Entry", "IoT Sensors (Simulated)", "Historical Data"])
        
        st.markdown("---")
        st.markdown("### 🔄 System Status")
        
        if st.button("Initialize AgroMind+", type="primary"):
            with st.spinner("Loading LSTM model..."):
                try:
                    st.session_state.system = AgroMindIntegratedSystem()
                    st.success("✅ System initialized successfully!")
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")
                    st.info("Please train the model first")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Input", "🌾 Crop Recommendations", 
                                       "💡 Advisory Report", "📈 Analytics"])
    
    with tab1:
        st.markdown("## 📝 Enter Environmental Data")
        st.markdown("Provide data for the last 4 weeks to get accurate predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌱 Soil Parameters")
            
            # Create 4-week data entry
            weeks_data = []
            for week in range(1, 5):
                with st.expander(f"Week {week} Data", expanded=(week==4)):
                    cols = st.columns(3)
                    
                    with cols[0]:
                        N = st.number_input(f"N (kg/ha)", min_value=0.0, max_value=350.0, 
                                          value=120.0-week*3, key=f"N_{week}")
                        P = st.number_input(f"P (kg/ha)", min_value=0.0, max_value=70.0, 
                                          value=40.0-week*1, key=f"P_{week}")
                        K = st.number_input(f"K (kg/ha)", min_value=0.0, max_value=70.0, 
                                          value=42.0-week*1, key=f"K_{week}")
                    
                    with cols[1]:
                        pH = st.number_input(f"pH", min_value=4.0, max_value=9.0, 
                                           value=6.8, key=f"pH_{week}")
                        temp = st.number_input(f"Temperature (°C)", min_value=10.0, 
                                             max_value=45.0, value=26.0+week*0.5, key=f"temp_{week}")
                        humidity = st.number_input(f"Humidity (%)", min_value=20.0, 
                                                  max_value=100.0, value=72.0+week*2, key=f"hum_{week}")
                    
                    with cols[2]:
                        moisture = st.number_input(f"Soil Moisture (%)", min_value=20.0, 
                                                  max_value=100.0, value=65.0+week*2, key=f"moist_{week}")
                        rainfall = st.number_input(f"Rainfall (mm)", min_value=0.0, 
                                                  max_value=300.0, value=80.0+week*5, key=f"rain_{week}")
                        sunlight = st.number_input(f"Sunlight (hrs/day)", min_value=0.0, 
                                                  max_value=14.0, value=6.5-week*0.2, key=f"sun_{week}")
                    
                    weeks_data.append([N, P, K, pH, temp, humidity, moisture, rainfall, sunlight])
        
        with col2:
            st.markdown("### 📈 Data Visualization")
            
            if len(weeks_data) == 4:
                # Create DataFrame
                df_viz = pd.DataFrame(weeks_data, 
                                     columns=['N', 'P', 'K', 'pH', 'Temp', 'Humidity', 
                                             'Moisture', 'Rainfall', 'Sunlight'])
                df_viz['Week'] = range(1, 5)
                
                # Plot trends
                st.line_chart(df_viz.set_index('Week')[['N', 'P', 'K']], height=200)
                st.caption("NPK Trends")
                
                st.line_chart(df_viz.set_index('Week')[['Temp', 'Humidity', 'Moisture']], height=200)
                st.caption("Climate Trends")
        
        # Predict button
        st.markdown("---")
        if st.button("🔮 Get Crop Recommendations", type="primary", use_container_width=True):
            if st.session_state.system is None:
                st.error("⚠️ Please initialize the system first (see sidebar)")
            else:
                with st.spinner("🤖 LSTM model analyzing data..."):
                    sequence_data = np.array(weeks_data)
                    recommendations = st.session_state.system.predict_top_crops(sequence_data, top_k=4)
                    st.session_state.recommendations = recommendations
                    st.success("✅ Analysis complete! Check 'Crop Recommendations' tab")
                    st.balloons()
    
    with tab2:
        st.markdown("## 🌾 Top-4 Crop Recommendations")
        
        if st.session_state.recommendations is None:
            st.info("👈 Please enter data and get recommendations first")
        else:
            recommendations = st.session_state.recommendations
            
            st.markdown("### 🏆 Your Personalized Recommendations")
            
            # Display recommendations
            for i, rec in enumerate(recommendations):
                rank_class = f"rank-{rec['rank']}"
                
                with st.container():
                    st.markdown(f'<div class="crop-card {rank_class}">', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
                    
                    with col1:
                        medal = ["🥇", "🥈", "🥉", "🏅"][i]
                        st.markdown(f"### {medal}")
                        st.markdown(f"**Rank {rec['rank']}**")
                    
                    with col2:
                        st.markdown(f"### {rec['crop']}")
                        st.metric("Suitability", rec['confidence'])
                    
                    with col3:
                        st.metric("PSI Score", f"{rec['psi_percentage']:.1f}%")
                        st.caption(f"Rating: {rec['psi_rating']}")
                    
                    with col4:
                        if st.button(f"Select {rec['crop']}", key=f"select_{i}", 
                                   type="primary" if i==0 else "secondary"):
                            st.session_state.selected_crop = rec
                            
                            # Generate advisory
                            with st.spinner("Generating advisory..."):
                                sequence_data = np.array(weeks_data)
                                current_conditions = {
                                    'N': sequence_data[-1, 0],
                                    'P': sequence_data[-1, 1],
                                    'K': sequence_data[-1, 2],
                                    'pH': sequence_data[-1, 3],
                                    'Temperature': sequence_data[-1, 4],
                                    'Humidity': sequence_data[-1, 5],
                                    'Moisture': sequence_data[-1, 6],
                                    'Rainfall': sequence_data[-1, 7],
                                    'Sunlight': sequence_data[-1, 8]
                                }
                                
                                advisory = st.session_state.system.generate_adaptive_advisory(
                                    rec, current_conditions, farm_size
                                )
                                st.session_state.advisory_report = advisory
                                st.success(f"✅ Advisory generated for {rec['crop']}!")
                                st.info("👉 Check 'Advisory Report' tab")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("")  # Spacing
    
    with tab3:
        st.markdown("## 💡 Comprehensive Advisory Report")
        
        if st.session_state.advisory_report is None:
            st.info("👈 Please select a crop first")
        else:
            report = st.session_state.advisory_report
            selected = st.session_state.selected_crop
            
            # Narrative
            st.markdown("### 📖 Explainable Crop Narrative")
            st.info(report['narrative'])
            
            # Fertilizer Plan
            st.markdown("### 💊 Fertilizer Recommendations")
            
            for fert in report['fertilizer_plan']['fertilizers']:
                with st.expander(f"🌿 {fert['name']}", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Quantity", f"{fert['quantity_kg']} kg")
                        st.caption(f"For {farm_size} hectares")
                    
                    with col2:
                        st.markdown(f"**Application:**")
                        st.write(fert['application'])
                        st.caption(f"Purpose: {fert['deficit_addressed']}")
            
            # Irrigation Plan
            st.markdown("### 💧 Irrigation Schedule")
            
            irr = report['irrigation_plan']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Irrigation Type", irr['recommended_type'])
            
            with col2:
                st.metric("Frequency", irr['frequency'])
            
            with col3:
                st.metric("Water Depth", irr['water_depth'])
            
            st.info(f"💧 Total Water Required: **{irr['total_water_required_mm']} mm**")
            
            if irr['adjustments']:
                st.warning("⚠️ **Weather-Based Adjustments:**")
                for adj in irr['adjustments']:
                    st.write(f"• {adj['action']} - {adj['reason']}")
            
            st.markdown("**Critical Irrigation Stages:**")
            for stage in irr['critical_stages']:
                st.write(f"• {stage}")
            
            # Yield Prediction
            st.markdown("### 📈 Yield Prediction")
            
            yp = report['yield_prediction']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Expected Yield", f"{yp['predicted_yield_t_ha']} t/ha")
            
            with col2:
                st.metric("Total Expected", f"{yp['predicted_yield_t_ha'] * farm_size:.1f} tons")
            
            with col3:
                st.metric("Confidence", yp['confidence'])
            
            # PSI Breakdown
            st.markdown("### 🌱 Sustainability Analysis")
            
            psi = report['psi']
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("PSI Score", f"{psi['psi_percentage']}%", psi['rating'])
            
            with col2:
                components = psi['components']
                st.progress(components['water_efficiency'])
                st.caption(f"Water Efficiency: {components['water_efficiency']*100:.0f}%")
                
                st.progress(components['fertilizer_efficiency'])
                st.caption(f"Fertilizer Efficiency: {components['fertilizer_efficiency']*100:.0f}%")
                
                st.progress(components['yield_potential'])
                st.caption(f"Yield Potential: {components['yield_potential']*100:.0f}%")
            
            # Download Report
            st.markdown("---")
            if st.button("📥 Download Full Report (PDF)", use_container_width=True):
                st.info("PDF generation feature coming soon!")
    
    with tab4:
        st.markdown("## 📊 Farm Analytics")
        
        st.info("🚧 Analytics dashboard under development")
        
        st.markdown("### Upcoming Features:")
        st.markdown("""
        - 📈 Historical yield tracking
        - 🌍 Regional comparison
        - 💰 Profit analysis
        - 🌦️ Weather forecasts
        - 📊 Crop rotation suggestions
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🌾 AgroMind+ | LSTM-Based Smart Crop Advisory System<br>
        Empowering Farmers with AI & Edge Computing
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()