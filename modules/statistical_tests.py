import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

def run():
    st.header("🔬 Statistical Tests")
    if st.session_state.data is not None:
        df = st.session_state.data.dropna() # Use dropna for statistical validity simply
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        tab1, tab2, tab3, tab4 = st.tabs(["Normality Tests", "Parametric Tests", "Non-Parametric Tests", "Chi-Square Test"])
        
        with tab1:
            st.subheader("Normality Tests (Shapiro-Wilk, Kolmogorov-Smirnov)")
            if numeric_cols:
                col = st.selectbox("Select numeric variable", numeric_cols, key="norm_col")
                test_type = st.radio("Test Type", ["Shapiro-Wilk", "Kolmogorov-Smirnov"])
                
                if st.button("Run Normality Test"):
                    if test_type == "Shapiro-Wilk":
                        stat, p = stats.shapiro(df[col])
                        st.write(f"**Statistic:** {stat:.4f}")
                        st.write(f"**p-value:** {p:.4g}")
                    elif test_type == "Kolmogorov-Smirnov":
                        stat, p = stats.kstest(df[col], 'norm', args=(df[col].mean(), df[col].std()))
                        st.write(f"**Statistic:** {stat:.4f}")
                        st.write(f"**p-value:** {p:.4g}")
                        
                    if p > 0.05:
                        st.success("Sample looks Gaussian (fail to reject H0).")
                    else:
                        st.error("Sample does not look Gaussian (reject H0).")
            else:
                st.info("No numeric columns available.")
                
        with tab2:
            st.subheader("Parametric Tests")
            st.info("Assumes normality and equal variance.")
            if len(numeric_cols) >= 1:
                test_type = st.selectbox("Parametric Test", ["Independent T-Test", "Paired T-Test", "One-Way ANOVA"])
                
                if test_type in ["Independent T-Test", "Paired T-Test"] and len(numeric_cols) >= 2:
                    col1 = st.selectbox("Group/Sample 1", numeric_cols, key="ptest_col1")
                    col2 = st.selectbox("Group/Sample 2", [c for c in numeric_cols if c != col1], key="ptest_col2")
                    
                    if st.button("Run T-Test"):
                        if test_type == "Independent T-Test":
                            stat, p = stats.ttest_ind(df[col1], df[col2], equal_var=False)
                        else:
                            stat, p = stats.ttest_rel(df[col1], df[col2])
                            
                        st.write(f"**T-Statistic:** {stat:.4f}")
                        st.write(f"**p-value:** {p:.4g}")
                        if p < 0.05:
                            st.error("Significant difference between groups (reject H0).")
                        else:
                            st.success("No significant difference between groups (fail to reject H0).")
                            
                elif test_type == "One-Way ANOVA" and numeric_cols and cat_cols:
                    target_col = st.selectbox("Numeric Target (Dependent Variable)", numeric_cols, key="anova_target")
                    group_col = st.selectbox("Categorical Group (Independent Variable)", cat_cols, key="anova_group")
                    
                    if st.button("Run ANOVA"):
                        groups = [group for name, group in df.groupby(group_col)[target_col]]
                        stat, p = stats.f_oneway(*groups)
                        st.write(f"**F-Statistic:** {stat:.4f}")
                        st.write(f"**p-value:** {p:.4g}")
                        if p < 0.05:
                            st.error("Significant difference between at least two group means (reject H0).")
                        else:
                            st.success("No significant difference found between group means (fail to reject H0).")
            else:
                st.info("Insufficient numeric columns for these tests.")
                
        with tab3:
            st.subheader("Non-Parametric Tests")
            st.info("Does not assume normality.")
            if len(numeric_cols) >= 1:
                np_test = st.selectbox("Non-Parametric Test", ["Mann-Whitney U Test", "Kruskal-Wallis H Test"])
                
                if np_test == "Mann-Whitney U Test" and len(numeric_cols) >= 2:
                    col1 = st.selectbox("Sample 1", numeric_cols, key="np_col1")
                    col2 = st.selectbox("Sample 2", [c for c in numeric_cols if c != col1], key="np_col2")
                    if st.button("Run Mann-Whitney", key="btn_mw"):
                        stat, p = stats.mannwhitneyu(df[col1], df[col2])
                        st.write(f"**U-Statistic:** {stat:.4f}")
                        st.write(f"**p-value:** {p:.4g}")
                        if p < 0.05:
                            st.error("Distributions are not equal (reject H0).")
                        else:
                            st.success("Distributions are equal (fail to reject H0).")
                            
                elif np_test == "Kruskal-Wallis H Test" and numeric_cols and cat_cols:
                    target_col = st.selectbox("Numeric Target", numeric_cols, key="kw_target")
                    group_col = st.selectbox("Categorical Group", cat_cols, key="kw_group")
                    
                    if st.button("Run Kruskal-Wallis", key="btn_kw"):
                        groups = [group for name, group in df.groupby(group_col)[target_col]]
                        stat, p = stats.kruskal(*groups)
                        st.write(f"**H-Statistic:** {stat:.4f}")
                        st.write(f"**p-value:** {p:.4g}")
                        if p < 0.05:
                            st.error("Significant difference between group medians (reject H0).")
                        else:
                            st.success("No significant difference found between group medians (fail to reject H0).")
            else:
                st.info("Not enough numeric columns.")
                
        with tab4:
            st.subheader("Chi-Square Test of Independence")
            if len(cat_cols) >= 2:
                cat1 = st.selectbox("Variable 1", cat_cols, key="chi_var1")
                cat2 = st.selectbox("Variable 2", [c for c in cat_cols if c != cat1], key="chi_var2")
                
                if st.button("Run Chi-Square Test"):
                    contingency = pd.crosstab(df[cat1], df[cat2])
                    st.write("Contingency Table:")
                    st.dataframe(contingency, use_container_width=True)
                    
                    stat, p, dof, expected = stats.chi2_contingency(contingency)
                    st.write(f"**Chi-Square Statistic:** {stat:.4f}")
                    st.write(f"**Degrees of Freedom:** {dof}")
                    st.write(f"**p-value:** {p:.4g}")
                    
                    if p < 0.05:
                        st.error(f"Variables '{cat1}' and '{cat2}' are dependent (reject H0).")
                    else:
                        st.success(f"Variables '{cat1}' and '{cat2}' are independent (fail to reject H0).")
            else:
                st.info("Need at least 2 categorical variables for Chi-Square Test of Independence.")

    else:
        st.warning("Please upload data first in the Data Import module.")
