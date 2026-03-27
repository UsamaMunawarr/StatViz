import streamlit as st
import pandas as pd
import numpy as np

def run():
    st.header("🗃️ Select Cases (Data Filter)")
    
    if st.session_state.data is not None:
        if 'original_data' not in st.session_state:
            st.session_state.original_data = st.session_state.data.copy()
            
        df = st.session_state.data.copy()
        
        st.markdown("Use this module to isolate or filter out specific portions of your dataset. All subsequent modules (EDA, Visualization, Models) will use this filtered subset.")
        
        col1, col2 = st.columns([2, 1])
        with col2:
            st.metric("Current Rows", f"{len(df):,}")
            if st.button("🔄 Reset All Filters (Revert to Original)"):
                st.session_state.data = st.session_state.original_data.copy()
                st.success("✅ Dataset restored to its original state.")
                st.rerun()

        with col1:
            st.subheader("Add Filter Condition")
            filter_col = st.selectbox("Select variable to filter by:", df.columns)
            
            # Determine column type safely
            if pd.api.types.is_numeric_dtype(df[filter_col]):
                min_val = float(df[filter_col].min())
                max_val = float(df[filter_col].max())
                
                # Check if values are essentially integers to format slider better
                if min_val.is_integer() and max_val.is_integer():
                    range_val = st.slider("Select numeric range to KEEP:", 
                                            min_value=int(min_val), 
                                            max_value=int(max_val), 
                                            value=(int(min_val), int(max_val)))
                else:
                    range_val = st.slider("Select numeric range to KEEP:", 
                                            min_value=min_val, 
                                            max_value=max_val, 
                                            value=(min_val, max_val))
                
                if st.button("✂️ Apply Numeric Filter"):
                    df_filtered = df[(df[filter_col] >= range_val[0]) & (df[filter_col] <= range_val[1])]
                    st.session_state.data = df_filtered
                    st.success(f"✅ Filter applied! Kept {len(df_filtered)} rows where {filter_col} is between {range_val[0]} and {range_val[1]}")
                    st.rerun()
                    
            elif pd.api.types.is_datetime64_any_dtype(df[filter_col]):
                min_date = df[filter_col].min().date()
                max_date = df[filter_col].max().date()
                
                date_range = st.date_input("Select date range to KEEP:", value=(min_date, max_date), min_value=min_date, max_value=max_date)
                
                if st.button("✂️ Apply Date Filter"):
                    if len(date_range) == 2:
                        start_date, end_date = date_range
                        df_filtered = df[(df[filter_col].dt.date >= start_date) & (df[filter_col].dt.date <= end_date)]
                        st.session_state.data = df_filtered
                        st.success(f"✅ Filter applied! Kept {len(df_filtered)} rows.")
                        st.rerun()
                    else:
                        st.warning("Please select a valid start and end date.")
                        
            else:
                # Categorical or Text
                unique_vals = df[filter_col].dropna().unique().tolist()
                
                if len(unique_vals) > 100:
                    st.warning("⚠️ This column has too many unique values to render a fast multiselect. Showing top 100 frequency values.")
                    unique_vals = df[filter_col].value_counts().head(100).index.tolist()

                selected_vals = st.multiselect("Select categories to KEEP:", unique_vals, default=unique_vals)
                
                if st.button("✂️ Apply Category Filter"):
                    df_filtered = df[df[filter_col].isin(selected_vals)]
                    st.session_state.data = df_filtered
                    st.success(f"✅ Filter applied! Kept {len(df_filtered)} rows matching your selection.")
                    st.rerun()
        
        st.divider()
        st.subheader("👀 Filtered Dataset Preview")
        st.dataframe(df.head(50), use_container_width=True)

    else:
        st.warning("Please upload data first in the Data Import module.")
