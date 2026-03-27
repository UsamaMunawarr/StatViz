import streamlit as st
import base64
import os
from utils.helpers import set_page_config
from streamlit_option_menu import option_menu
from modules import data_import, datatype, data_filtering, eda, descriptive, visualization, statistical_tests, modeling, report_gen, time_series, multivariate, contact

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None

def main():
    set_page_config()
    
    # Custom CSS to match the given style requests and dark theme
    st.markdown("""
    <style>
        .main {
            background-color: #0a1628;
        }
        .main-title {
            text-align: center;
            font-size: 42px;
            font-weight: bold;
            color: #00d4aa;
            font-family: 'Segoe UI', sans-serif;
            margin-top: 10px;
            margin-bottom: 10px;
        }
        h2 {
            color: #00d4aa !important;
            padding-bottom: 10px;
            border-bottom: 2px solid #112240;
        }
        h3 {
            color: #48C9B0 !important;
        }
        [data-testid="stMetricValue"] {
            color: #00d4aa !important;
        }
        .feature-card {
            background-color: #112240;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            height: 190px;
            border-left: 5px solid #00d4aa;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 15px rgba(0, 212, 170, 0.2);
            border-left: 5px solid #1abc9c;
        }
        .hero-title {
            font-size: 55px;
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, #00d4aa, #48C9B0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
            padding-bottom: 0;
            line-height: 1.2;
        }
        .hero-subtitle {
            font-size: 24px;
            color: #a9a9a9;
            margin-top: 5px;
            margin-bottom: 20px;
        }
        .feature-title {
            color: #ffffff;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .feature-text {
            color: #a9a9a9;
            font-size: 14px;
            line-height: 1.5;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Top Navigation Menu
    selected = option_menu(
        menu_title=None,
        options=["Home", "Import", "Prep", "Filter", "EDA", "Stats", "Plot", "Tests", "Multivariate", "Models", "TS", "PDF", "Contact"],
        icons=["house", "cloud-upload", "shuffle", "funnel-fill", "search", "table", "bar-chart", "clipboard-data", "layers", "robot", "graph-up", "file-earmark-pdf", "envelope"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#112240", "max-width": "100%", "overflow": "auto"},
            "icon": {"color": "#ffffff", "font-size": "24px"},
            "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "padding": "10px", "--hover-color": "#2c3e50", "display": "flex", "flex-direction": "column", "align-items": "center"},
            "nav-link-selected": {"background-color": "#00d4aa", "color": "#112240"}, 
        },
    )

    # Inject extra CSS to change the icon color when selected, as option_menu doesn't expose it directly
    st.markdown("""
    <style>
        .nav-link.active i {
            color: #112240 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    if selected == "Home":
        st.write("")
        col_hero1, col_hero2 = st.columns([1.5, 1])
        
        with col_hero1:
            st.markdown('<div class="hero-title">StatViz</div>', unsafe_allow_html=True)
            st.markdown('<div class="hero-subtitle">Your Free, Browser-Based Web App</div>', unsafe_allow_html=True)
            st.markdown("<p style='font-size: 16px; color: #d3d3d3; margin-bottom: 25px; line-height: 1.6;'>Analyze, visualize, and model your datasets effortlessly. Designed explicitly for Researchers, Data Scientists, and Scholars to eliminate the friction of modern data analysis without writing a single line of code.</p>", unsafe_allow_html=True)
            
            st.markdown("<p style='color:#a9a9a9;'>Developed with passion by <b>Usama Munawar</b></p>", unsafe_allow_html=True)

        with col_hero2:
            if os.path.exists("developer.jpeg"):
                try:
                    # Using custom HTML for beautifully rounded profile picture with a glowing border
                    img_bytes = open("developer.jpeg", "rb").read()
                    encoded = base64.b64encode(img_bytes).decode()
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
                            <img src="data:image/jpeg;base64,{encoded}" style="border-radius: 50%; max-width: 280px; width: 100%; aspect-ratio: 1/1; border: 4px solid #00d4aa; object-fit: cover; box-shadow: 0 10px 25px rgba(0,212,170,0.3);">
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                except Exception:
                    st.image("developer.jpeg", use_container_width=True)
                    
        st.write("---")
        st.markdown("<h3 style='text-align: center; margin-top: 20px; margin-bottom: 20px;'>🏗️ Platform Features & Architecture</h3>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="feature-card" style="border-left-color: #00d4aa;">
                <div class="feature-title">📂 1. Import & Prep</div>
                <div class="feature-text">Upload datasets (CSV, Excel), transform numeric/categorical types, and apply elite Z-Score or Min-Max scaling.</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="feature-card" style="border-left-color: #00d4aa;">
                <div class="feature-title">🧠 4. Statistical Tests</div>
                <div class="feature-text">Run comprehensive hypothesis testing including Normality, Parametric (ANOVA, T-Tests), and Non-Parametric tests instantly.</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="feature-card" style="border-left-color: #48C9B0;">
                <div class="feature-title">🔀 2. Filter & EDA</div>
                <div class="feature-text">Select cases dynamically, handle missing values (Heatmaps), and eliminate outliers with advanced Winsorization.</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="feature-card" style="border-left-color: #48C9B0;">
                <div class="feature-title">🤖 5. Machine Learning</div>
                <div class="feature-text">Train models for Regression/Classification with custom Hyperparameter Tuning and automated PCA pre-processing.</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="feature-card" style="border-left-color: #1abc9c;">
                <div class="feature-title">📊 3. Plot & Multivariate</div>
                <div class="feature-text">Build interactive charts and perform advanced Dimensionality Reduction (PCA) or Factor Analysis (EFA).</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="feature-card" style="border-left-color: #1abc9c;">
                <div class="feature-title">🖨️ 6. Automation & Reports</div>
                <div class="feature-text">Automate your Time-Series analysis (ARIMA) and generate fully compiled PDF Statistical Reports on the fly.</div>
            </div>
            """, unsafe_allow_html=True)

    elif selected == "Import":
        data_import.run()
    elif selected == "Prep":
        datatype.run()
    elif selected == "Filter":
        data_filtering.run()
    elif selected == "EDA":
        eda.run()
    elif selected == "Stats":
        descriptive.run()
    elif selected == "Plot":
        visualization.run()
    elif selected == "Tests":
        statistical_tests.run()
    elif selected == "Multivariate":
        multivariate.run()
    elif selected == "Models":
        modeling.run()
    elif selected == "TS":
        time_series.run()
    elif selected == "PDF":
        report_gen.run()
    elif selected == "Contact":
        contact.run()
        
    # App-wide Footer
    st.markdown("""
    <div style='text-align: center; color: #00d4aa; font-size: 20px; margin-top: 50px;'>
    💚 Thank you for using StatViz Pro, share it with your friends! 😇 <br>
    <b>Developed by Usama Munawar</b>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
