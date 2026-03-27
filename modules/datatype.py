import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

def run():
    st.header("🔄 Data Type Management & Transformation")
    if st.session_state.data is not None:
        df = st.session_state.data
        
        st.subheader("Current Data Types")
        dtypes_df = df.dtypes.astype(str).reset_index().rename(columns={'index': 'Column Name', 0: 'Data Type'})
        st.dataframe(dtypes_df, use_container_width=True)

        st.divider()

        # Tool selector dropdown to keep the interface clean
        action = st.selectbox("Select Transformation Tool", [
            "Data Type Conversion", 
            "Categorical Variable Encoding", 
            "Data Scaling & Normalization",
            "Column Renaming", 
            "Delete Column"
        ])

        if action == "Data Type Conversion":
            st.subheader("Convert Data Type")
            col_to_convert = st.selectbox("Select column to convert", df.columns)
            new_type = st.selectbox("Select new data type", ["Numeric (float)", "Numeric (int)", "Categorical (string/object)", "Boolean", "DateTime"])

            if st.button("Convert Data Type"):
                try:
                    if new_type == "Numeric (float)":
                        df[col_to_convert] = pd.to_numeric(df[col_to_convert], errors='coerce').astype(float)
                    elif new_type == "Numeric (int)":
                        df[col_to_convert] = pd.to_numeric(df[col_to_convert], errors='coerce').astype('Int64')
                    elif new_type == "Categorical (string/object)":
                        df[col_to_convert] = df[col_to_convert].astype(str)
                    elif new_type == "Boolean":
                        df[col_to_convert] = df[col_to_convert].astype(bool)
                    elif new_type == "DateTime":
                        df[col_to_convert] = pd.to_datetime(df[col_to_convert], errors='coerce')
                    
                    st.session_state.data = df
                    st.success(f"Successfully converted '{col_to_convert}' to {new_type}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error converting data type: {e}")

        elif action == "Categorical Variable Encoding":
            st.subheader("Categorical Variable Encoding")
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                col_to_encode = st.selectbox("Select column to encode", cat_cols)
                encoding_type = st.radio("Encoding Method", ["Label Encoding", "One-Hot Encoding"])
                
                if st.button("Apply Encoding"):
                    if encoding_type == "Label Encoding":
                        le = LabelEncoder()
                        df[f"{col_to_encode}_encoded"] = le.fit_transform(df[col_to_encode].astype(str))
                        st.success(f"Label encoded '{col_to_encode}' into new column '{col_to_encode}_encoded'")
                        
                    elif encoding_type == "One-Hot Encoding":
                        dummies = pd.get_dummies(df[col_to_encode], prefix=col_to_encode, drop_first=False)
                        df = pd.concat([df, dummies], axis=1)
                        st.success(f"One-Hot encoded '{col_to_encode}' (added {dummies.shape[1]} new columns)")
                        
                    st.session_state.data = df
                    st.rerun()
            else:
                st.info("No categorical columns detected.")

        elif action == "Data Scaling & Normalization":
            st.subheader("Numeric Scaling & Normalization")
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            if num_cols:
                col_to_scale = st.selectbox("Select column to scale", num_cols)
                scaling_method = st.radio("Scaling Method", ["Standardization (Z-score)", "Min-Max Scaling (0-1)"])
                
                if st.button("Apply Scaling"):
                    if scaling_method == "Standardization (Z-score)":
                        scaler = StandardScaler()
                        df[f"{col_to_scale}_scaled"] = scaler.fit_transform(df[[col_to_scale]])
                        st.success(f"Z-score standardized '{col_to_scale}' into new column '{col_to_scale}_scaled'")
                    elif scaling_method == "Min-Max Scaling (0-1)":
                        scaler = MinMaxScaler()
                        df[f"{col_to_scale}_normalized"] = scaler.fit_transform(df[[col_to_scale]])
                        st.success(f"Min-Max normalized '{col_to_scale}' into new column '{col_to_scale}_normalized'")
                    
                    st.session_state.data = df
                    st.rerun()
            else:
                st.info("No numeric columns detected for scaling.")

        elif action == "Column Renaming":
            st.subheader("Rename Column")
            col_to_rename = st.selectbox("Select column to rename", df.columns)
            new_col_name = st.text_input("New column name")
            
            if st.button("Rename Column"):
                if new_col_name and new_col_name not in df.columns:
                    df = df.rename(columns={col_to_rename: new_col_name})
                    st.session_state.data = df
                    st.success(f"Renamed '{col_to_rename}' to '{new_col_name}'")
                    st.rerun()
                elif new_col_name in df.columns:
                    st.error("Column name already exists!")
                else:
                    st.warning("Please enter a new column name.")

        elif action == "Delete Column":
            st.subheader("Delete Column")
            col_to_delete = st.selectbox("Select column to delete", df.columns)
            
            if st.button("Delete Column"):
                df = df.drop(columns=[col_to_delete])
                st.session_state.data = df
                st.success(f"Successfully deleted column '{col_to_delete}'.")
                st.rerun()

        st.divider()

        # Section to download Dataset
        st.subheader("💾 Save Transformed Dataset")
        st.info("Your changes are saved automatically in the app session. You can safely proceed to the EDA module or other tabs right now. Alternatively, you can download a copy of the transformed dataset below.")
        
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="transformed_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        with col2:
            try:
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                excel_data = buffer.getvalue()
                
                st.download_button(
                    label="Download as Excel",
                    data=excel_data,
                    file_name="transformed_dataset.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception as e:
                st.error("Excel download not available (openpyxl might be missing).")

    else:
        st.warning("Please upload data first in the Data Import module.")
