import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def run():
    st.header("📊 Descriptive Statistics")
    if st.session_state.data is not None:
        df = st.session_state.data
        
        tab1, tab2, tab3, tab4 = st.tabs(["Summary Statistics", "Distribution Metrics", "Frequency Tables", "Correlation Analysis"])
        
        with tab1:
            st.subheader("Numeric Summary Statistics")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                numeric_df = df[numeric_cols]
                desc_df = numeric_df.describe().transpose()
                st.dataframe(desc_df, use_container_width=True)
            else:
                st.info("No numeric columns found.")
                
        with tab2:
            st.subheader("Skewness, Kurtosis, & Variance")
            if numeric_cols:
                skew = numeric_df.skew()
                kurtosis = numeric_df.kurt()
                var = numeric_df.var()
                std = numeric_df.std()
                
                dist_df = pd.DataFrame({
                    "Variance": var,
                    "Std. Deviation": std,
                    "Skewness": skew,
                    "Kurtosis": kurtosis
                })
                st.dataframe(dist_df, use_container_width=True)
            else:
                st.info("No numeric columns found.")
                
        with tab3:
            st.subheader("Frequency Tables (Categorical Variables)")
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                col = st.selectbox("Select variable for frequency table", cat_cols)
                freq_df = df[col].value_counts().reset_index()
                freq_df.columns = [col, 'Count']
                freq_df['Percentage (%)'] = (freq_df['Count'] / freq_df['Count'].sum()) * 100
                st.dataframe(freq_df.style.format({'Percentage (%)': '{:.2f}%'}), use_container_width=True)
            else:
                st.info("No categorical columns found.")
                
        with tab4:
            st.subheader("Correlation Matrix")
            if len(numeric_cols) > 1:
                method = st.selectbox("Select Method", ["pearson", "spearman", "kendall"])
                corr = numeric_df.corr(method=method)
                
                fig = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis", aspect="auto",
                                title=f"Correlation Matrix ({method.capitalize()})")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least two numeric columns for correlation analysis.")
                
        st.divider()
        st.subheader("Export Statistics")
        try:
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                if numeric_cols:
                    desc_df.to_excel(writer, sheet_name="Summary Data")
                    dist_df.to_excel(writer, sheet_name="Distribution Metrics")
                df.describe(include='all').to_excel(writer, sheet_name="All Types Summary")
            st.download_button(
                label="Download Summary as Excel",
                data=buffer.getvalue(),
                file_name="descriptive_stats.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"Excel export error: {e}")

    else:
        st.warning("Please upload data first in the Data Import module.")
