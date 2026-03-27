import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def run():
    st.header("🔍 Exploratory Data Analysis (EDA)")
    if st.session_state.data is not None:
        df = st.session_state.data.copy()
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Missing Values", "Duplicates", "Outliers", "Download Cleaned Data"])
        
        with tab1:
            st.subheader("Dataset Overview")
            
            # Using metrics for eye-catching important figures
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", f"{df.shape[0]:,}")
            c2.metric("Columns", f"{df.shape[1]:,}")
            c3.metric("Total Missing Cells", f"{df.isnull().sum().sum():,}")
            
            st.divider()
            
            st.subheader("🛠️ Column Data Types")
            dtypes_df = df.dtypes.astype(str).reset_index().rename(columns={'index': 'Column', 0: 'Type'})
            st.dataframe(dtypes_df, use_container_width=True)
            
            st.subheader("👀 First 10 Rows")
            st.dataframe(df.head(10), use_container_width=True)

        with tab2:
            st.subheader("Missing Values Analysis")
            missing_stats = df.isnull().sum()
            missing_percent = (df.isnull().sum() / len(df)) * 100
            missing_df = pd.DataFrame({'Missing Values': missing_stats, 'Percentage (%)': missing_percent})
            missing_df = missing_df[missing_df['Missing Values'] > 0]
            
            if not missing_df.empty:
                st.dataframe(missing_df.style.format({'Percentage (%)': '{:.2f}%'}))
                
                # Missing Values Heatmap
                st.markdown("<h3 style='color:#00d4aa; margin-top: 20px;'>Missing Values Heatmap</h3>", unsafe_allow_html=True)
                fig = px.imshow(df.isnull().transpose(), color_continuous_scale='Blues')
                fig.update_layout(xaxis_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                st.subheader("Imputation Options")
                col_to_impute = st.selectbox("Select column to impute", missing_df.index)
                method = st.selectbox("Imputation Method", ["Mean", "Median", "Mode", "Forward Fill", "Backward Fill", "Drop Column", "Drop NaNs in Column", "Custom Value"])
                
                if method == "Custom Value":
                    custom_val = st.text_input("Enter custom value")
                else:
                    custom_val = None
                    
                if st.button("Apply Imputation"):
                    if method == "Mean":
                        df[col_to_impute] = df[col_to_impute].fillna(df[col_to_impute].mean())
                    elif method == "Median":
                        df[col_to_impute] = df[col_to_impute].fillna(df[col_to_impute].median())
                    elif method == "Mode":
                        mode_val = df[col_to_impute].mode()
                        if not mode_val.empty:
                            df[col_to_impute] = df[col_to_impute].fillna(mode_val[0])
                    elif method == "Forward Fill":
                        df[col_to_impute] = df[col_to_impute].ffill()
                    elif method == "Backward Fill":
                        df[col_to_impute] = df[col_to_impute].bfill()
                    elif method == "Drop Column":
                        df = df.drop(columns=[col_to_impute])
                    elif method == "Drop NaNs in Column":
                        df = df.dropna(subset=[col_to_impute])
                    elif method == "Custom Value" and custom_val is not None:
                        df[col_to_impute] = df[col_to_impute].fillna(custom_val)
                    
                    st.session_state.data = df
                    st.success(f"✅ Successfully applied {method} imputation on '{col_to_impute}'")
                    st.rerun()
            else:
                st.success("🎉 Incredible! No missing values found in the dataset.")
                
        with tab3:
            st.subheader("Duplicates Detection")
            num_duplicates = df.duplicated().sum()
            
            c1, c2 = st.columns([1, 2])
            c1.metric("Duplicate Rows", f"{num_duplicates:,}")
            
            with c2:
                if num_duplicates > 0:
                    st.info("💡 Having duplicate rows might skew your models. You can easily remove them below.")
                    if st.button("🗑️ Remove Duplicate Rows", use_container_width=True):
                        st.session_state.data = df.drop_duplicates()
                        st.success(f"Removed {num_duplicates} duplicate rows successfully!")
                        st.rerun()
                else:
                    st.success("Clean dataset! No duplicate rows found.")

        with tab4:
            st.subheader("Outlier Detection & Treatment")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                outlier_col = st.selectbox("Select column to check for outliers", numeric_cols)
                
                method = st.radio("Detection Method", ["IQR Method", "Z-Score Method"], horizontal=True)
                
                if method == "IQR Method":
                    # Calculate IQR
                    Q1 = df[outlier_col].quantile(0.25)
                    Q3 = df[outlier_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                else:
                    # Calculate Z-Score bounds
                    z_thresh = st.slider("Z-Score Threshold (Standard Deviations)", 1.0, 5.0, 3.0, 0.5)
                    mean_val = df[outlier_col].mean()
                    std_val = df[outlier_col].std()
                    lower_bound = mean_val - z_thresh * std_val
                    upper_bound = mean_val + z_thresh * std_val
                
                outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]
                
                st.metric(f"Outliers computed in '{outlier_col}'", f"{len(outliers):,}")
                
                if not outliers.empty:
                    # Plot Boxplot
                    fig = px.box(df, y=outlier_col, title=f"Boxplot of {outlier_col}", points="outliers", color_discrete_sequence=['#00d4aa'])
                    
                    # Add min/max threshold lines visually to help
                    fig.add_shape(type='line', y0=lower_bound, y1=lower_bound, x0=-0.5, x1=0.5, line=dict(color='red', dash='dash'), name='Lower Bound')
                    fig.add_shape(type='line', y0=upper_bound, y1=upper_bound, x0=-0.5, x1=0.5, line=dict(color='red', dash='dash'), name='Upper Bound')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("<h3 style='color:#00d4aa;'>Action Strategy</h3>", unsafe_allow_html=True)
                    outlier_action = st.radio("Choose what to do with these outliers:", ["None", "Remove Outliers", "Cap Outliers (Winsorize to Bounds)", "Cap Outliers (Winsorize to Custom Min/Max)"])
                    
                    custom_min = lower_bound
                    custom_max = upper_bound
                    
                    if outlier_action == "Cap Outliers (Winsorize to Custom Min/Max)":
                        c1, c2 = st.columns(2)
                        with c1:
                            custom_min = st.number_input("Custom Minimum Cap", value=float(lower_bound))
                        with c2:
                            custom_max = st.number_input("Custom Maximum Cap", value=float(upper_bound))
                    
                    if st.button("Apply Outlier Action"):
                        if outlier_action == "Remove Outliers":
                            st.session_state.data = df[(df[outlier_col] >= lower_bound) & (df[outlier_col] <= upper_bound)]
                            st.success(f"✅ Removed {len(outliers)} outliers from '{outlier_col}'")
                            st.rerun()
                        elif outlier_action == "Cap Outliers (Winsorize to Bounds)":
                            df[outlier_col] = np.where(df[outlier_col] > upper_bound, upper_bound,
                                                np.where(df[outlier_col] < lower_bound, lower_bound, df[outlier_col]))
                            st.session_state.data = df
                            st.success(f"✅ Capped {len(outliers)} outliers in '{outlier_col}' to calculated limits")
                            st.rerun()
                        elif outlier_action == "Cap Outliers (Winsorize to Custom Min/Max)":
                            df[outlier_col] = np.where(df[outlier_col] > custom_max, custom_max,
                                                np.where(df[outlier_col] < custom_min, custom_min, df[outlier_col]))
                            st.session_state.data = df
                            st.success(f"✅ Capped outliers to Min: {custom_min:.2f} and Max: {custom_max:.2f}")
                            st.rerun()
            else:
                st.info("No numerical columns available for outlier detection.")
                
        with tab5:
            st.subheader("Download Cleaned Dataset")
            st.info("Your updated manipulations to this dataset (imputations, duplicate removal, outlier handling) have been automatically saved! You can also download a hard copy below.")
            
            c1, c2 = st.columns(2)
            csv = df.to_csv(index=False).encode('utf-8')
            c1.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            try:
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                excel_data = buffer.getvalue()
                
                c2.download_button(
                    label="📥 Download as Excel",
                    data=excel_data,
                    file_name="cleaned_dataset.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception as e:
                st.error("Excel download not available (openpyxl might be missing).")

    else:
        st.warning("Please upload data first in the Data Import module.")
