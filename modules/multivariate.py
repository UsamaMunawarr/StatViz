import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Check if factor_analyzer is available and patch for sklearn >= 1.3 compatibility
try:
    from factor_analyzer import FactorAnalyzer
    import factor_analyzer.factor_analyzer
    
    original_check_array = factor_analyzer.factor_analyzer.check_array
    
    def patched_check_array(*args, **kwargs):
        if 'force_all_finite' in kwargs:
            kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
        return original_check_array(*args, **kwargs)
        
    factor_analyzer.factor_analyzer.check_array = patched_check_array
    FACTOR_ANALYZER_AVAILABLE = True
except ImportError:
    FACTOR_ANALYZER_AVAILABLE = False

def run():
    st.header("🧬 Multivariate Analysis")
    st.write("Analyze patterns in multiple variables simultaneously. Use these tools to identify underlying structures (factors) or reduce data dimensionality (PCA).")
    
    if st.session_state.data is not None:
        df = st.session_state.data.dropna() # Multivariate analysis requires no NaNs
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("Multivariate analysis requires at least 2 numeric variables.")
            return
            
        tab1, tab2 = st.tabs(["PCA (Principal Component Analysis)", "Factor Analysis (EFA)"])
        
        with tab1:
            st.subheader("Principal Component Analysis")
            st.markdown("PCA reduces the complexity of your data by creating new 'Principal Components' that capture the maximum variance. Great for visualization and simplifying models.")
            
            selected_pca_cols = st.multiselect("Select variables for PCA:", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
            
            if len(selected_pca_cols) >= 2:
                # Execution
                X = df[selected_pca_cols]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                n_comp = st.slider("Number of components to extract:", 2, min(len(selected_pca_cols), 10), 2)
                
                if st.button("🚀 Run PCA"):
                    pca = PCA(n_components=n_comp)
                    pca_result = pca.fit_transform(X_scaled)
                    
                    # Explained Variance
                    exp_var = pca.explained_variance_ratio_ * 100
                    st.success(f"PCA completed! Total variance explained by {n_comp} components: {sum(exp_var):.2f}%")
                    
                    # Explained Variance Chart
                    var_df = pd.DataFrame({'PC': [f'PC{i+1}' for i in range(len(exp_var))], 'Variance (%)': exp_var})
                    fig_var = px.bar(var_df, x='PC', y='Variance (%)', title="Explained Variance by Component", color_discrete_sequence=['#00d4aa'])
                    st.plotly_chart(fig_var, use_container_width=True)
                    
                    # PCA Visualization (Scatter)
                    if n_comp >= 2:
                        res_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_comp)])
                        
                        # Add a categorical column for coloring if it exists
                        cat_cols = st.session_state.data.select_dtypes(include=['object', 'category']).columns.tolist()
                        color_col = None
                        if cat_cols:
                            color_col = st.selectbox("Select variable to color by:", [None] + cat_cols)
                        
                        if color_col:
                            res_df[color_col] = df[color_col].values
                            
                        fig_scatter = px.scatter(res_df, x='PC1', y='PC2', color=color_col, title="PCA: Component 1 vs Component 2", template="plotly_dark")
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Factor Loadings
                    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_comp)], index=selected_pca_cols)
                    st.write("### 🏗️ Component Loadings (Structure Matrix)")
                    st.write("See which original variables contribute most to each Principal Component.")
                    st.dataframe(loadings.style.background_gradient(cmap='coolwarm'), use_container_width=True)
                    
            else:
                st.info("Select at least 2 variables.")

        with tab2:
            st.subheader("Exploratory Factor Analysis (EFA)")
            st.write("Identify latent factors that explain the relationships between your variables. Commonly used in psychological and social research.")
            
            if not FACTOR_ANALYZER_AVAILABLE:
                st.error("The `factor-analyzer` library is required for this module. Please install it or use PCA instead.")
                return
                
            selected_fa_cols = st.multiselect("Select variables for Factor Analysis:", numeric_cols, default=numeric_cols[:min(8, len(numeric_cols))], key="fa_cols")
            
            if len(selected_fa_cols) >= 3:
                n_factors = st.slider("Number of Factors to extract:", 1, min(len(selected_fa_cols)-1, 10), 2)
                rotation = st.selectbox("Rotation Method", ["varimax", "promax", "oblimin", "None"])
                rotation = None if rotation == "None" else rotation
                
                if st.button("🚀 Run Factor Analysis"):
                    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
                    fa.fit(df[selected_fa_cols])
                    
                    # Eigenvalues Table
                    ev, v = fa.get_eigenvalues()
                    st.write("### 📐 Eigenvalues")
                    ev_df = pd.DataFrame({"Factor": range(1, len(ev)+1), "Eigenvalue": ev})
                    st.dataframe(ev_df.head(10), use_container_width=True)
                    
                    # Scree Plot
                    fig_scree = px.line(ev_df, x='Factor', y='Eigenvalue', markers=True, title="Scree Plot")
                    fig_scree.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Kaiser Criterion (v=1)")
                    st.plotly_chart(fig_scree, use_container_width=True)
                    
                    # Loadings Matrix
                    loadings = pd.DataFrame(fa.loadings_, columns=[f'Factor {i+1}' for i in range(n_factors)], index=selected_fa_cols)
                    st.write("### 🏗️ Factor Loadings Matrix")
                    st.dataframe(loadings.style.background_gradient(cmap='RdYlGn'), use_container_width=True)
                    
                    # Variance Accounted for
                    var_df = pd.DataFrame(fa.get_factor_variance(), index=['SS Loadings', 'Proportion Var', 'Cumulative Var'], columns=[f'Factor {i+1}' for i in range(n_factors)])
                    st.write("### 📊 Variance Explained")
                    st.dataframe(var_df, use_container_width=True)
            else:
                st.info("Select at least 3 variables for EFA.")
                
    else:
        st.warning("Please upload data first in the Data Import module.")
