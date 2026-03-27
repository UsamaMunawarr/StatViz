import streamlit as st
import pandas as pd
import seaborn as sns

def run():
    st.header("📂 Data Import & Editor")
    
    st.markdown("### 1. Load Data")
    tab1, tab2 = st.tabs(["📤 Upload Your Own Dataset", "🔮 Use a Sample Dataset"])
    
    with tab1:
        st.write("Upload your personal dataset here (CSV, Excel, JSON or TXT).")
        uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx", "xls", "json", "txt"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            try:
                name = uploaded_file.name
                if name.endswith('.csv') or name.endswith('.txt'):
                    df = pd.read_csv(uploaded_file)
                elif name.endswith('.xlsx') or name.endswith('.xls'):
                    df = pd.read_excel(uploaded_file)
                elif name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                
                st.session_state.data = df
                st.session_state.original_data = df.copy()
                st.success("✅ File uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                
    with tab2:
        st.write("Don't have a dataset ready? Try one of our standard datasets to explore the app's features.")
        sample_choice = st.selectbox("Select a Sample Dataset", ["(Select one)", "Titanic (Survival Data)", "Iris (Flower Data)", "Penguins (Species Data)"])
        
        if st.button("Load Sample Dataset"):
            if sample_choice == "Titanic (Survival Data)":
                st.session_state.data = sns.load_dataset("titanic")
                st.session_state.original_data = st.session_state.data.copy()
                st.success("✅ Titanic dataset loaded successfully!")
            elif sample_choice == "Iris (Flower Data)":
                st.session_state.data = sns.load_dataset("iris")
                st.session_state.original_data = st.session_state.data.copy()
                st.success("✅ Iris dataset loaded successfully!")
            elif sample_choice == "Penguins (Species Data)":
                st.session_state.data = sns.load_dataset("penguins")
                st.session_state.original_data = st.session_state.data.copy()
                st.success("✅ Penguins dataset loaded successfully!")
            else:
                st.warning("Please select a valid sample dataset.")
            
    if st.session_state.data is not None:
        st.divider()
        st.subheader("📊 Dataset Shape")
        
        # Display dataset shape creatively with metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", f"{st.session_state.data.shape[0]:,}")
        col2.metric("Total Columns", f"{st.session_state.data.shape[1]:,}")
        col3.metric("Missing Values", f"{st.session_state.data.isnull().sum().sum():,}")
        
        st.divider()
        st.subheader("📝 Interactive Data Editor")
        st.info("💡 You can directly double-click any cell below to edit its value! You can also check the box on the left to delete rows. Changes are saved automatically in your session.")
        
        edited_df = st.data_editor(
            st.session_state.data, 
            num_rows="dynamic", # allow adding/deleting rows
            use_container_width=True
        )
        
        # Save the edited dataset back into the session memory
        st.session_state.data = edited_df

    else:
        st.info("ℹ️ Please upload or select a file to begin.")
