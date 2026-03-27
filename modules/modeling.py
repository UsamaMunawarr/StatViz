import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pickle
import io

def run():
    st.header("🤖 Statitics and Machine Learning Modeling")
    if st.session_state.data is not None:
        df = st.session_state.data.dropna() # Modeling requires no missing values
        
        tab1, tab2, tab3 = st.tabs(["Regression", "Classification", "Clustering (K-Means)"])
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        with tab1:
            st.subheader("Regression Models")
            model_type = st.selectbox("Select Regression Model", ["Simple Linear Regression", "Multiple Linear Regression", "Polynomial Regression"])
            
            target_col = st.selectbox("Target Variable (Y)", numeric_cols, key="reg_target")
            feature_cols = st.multiselect("Feature Variables (X)", [c for c in numeric_cols if c != target_col], key="reg_features")
            test_size = st.slider("Test Size (%)", 10, 50, 20, key="reg_test_size") / 100
            
            if st.button("Run Regression Model"):
                if target_col and feature_cols:
                    if model_type == "Simple Linear Regression" and len(feature_cols) > 1:
                        st.error("Simple Linear Regression requires only 1 feature variable.")
                    else:
                        X = df[feature_cols]
                        y = df[target_col]
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                        
                        # Hyperparameter Tuning
                        with st.expander("⚙️ Model Parameters (Advanced)", expanded=False):
                            if model_type == "Polynomial Regression":
                                degree = st.slider("Polynomial Degree", 2, 5, 2)
                                poly = PolynomialFeatures(degree=degree)
                                X_train = poly.fit_transform(X_train)
                                X_test = poly.transform(X_test)
                            
                            fit_intercept = st.toggle("Fit Intercept", value=True)
                        
                        # PCA option
                        use_pca = st.checkbox("🔮 Apply PCA (Dimensionality Reduction)", value=False, key="reg_pca")
                        if use_pca:
                            n_comp = st.number_input("PCA Components", 1, len(feature_cols), min(2, len(feature_cols)))
                            pca = PCA(n_components=n_comp)
                            X_train = pca.fit_transform(X_train)
                            X_test = pca.transform(X_test)

                        model = LinearRegression(fit_intercept=fit_intercept)
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        
                        # Metrics Dashboard
                        mse = mean_squared_error(y_test, predictions)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, predictions)
                        r2 = r2_score(y_test, predictions)
                        
                        st.divider()
                        st.subheader("🏆 Regression Metrics Dashboard")
                        cols = st.columns(4)
                        cols[0].metric("R² Score", f"{r2:.4f}")
                        cols[1].metric("MSE", f"{mse:.4f}")
                        cols[2].metric("RMSE", f"{rmse:.4f}")
                        cols[3].metric("MAE", f"{mae:.4f}")
                        
                        st.markdown("### 📈 Analysis Plot")
                        fig = px.scatter(x=y_test, y=predictions, labels={'x': 'True Values', 'y': 'Predicted Values'}, title="True vs Predicted")
                        fig.add_shape(type="line", line=dict(dash='dash', color='red'), x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max())
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.divider()
                        st.markdown("### 💾 Export Trained Model")
                        model_bytes = pickle.dumps(model)
                        st.download_button(
                            label="📥 Download Trained Model (.pkl)",
                            data=model_bytes,
                            file_name=f"{model_type.replace(' ', '_').lower()}.pkl",
                            mime="application/octet-stream",
                            use_container_width=True
                        )
                else:
                    st.warning("Please select target and at least one feature variable.")
                    
        with tab2:
            st.subheader("Classification Models")
            clf_type = st.selectbox("Select Classification Model", ["Logistic Regression", "K-Nearest Neighbors (KNN)", "Decision Tree", "Random Forest", "SVM", "XGBoost"])
            
            target_col = st.selectbox("Target Class (Y)", [c for c in df.columns if df[c].nunique() < 20], key="clf_target")
            feature_cols = st.multiselect("Feature Variables (X)", [c for c in numeric_cols if c != target_col], key="clf_features")
            test_size = st.slider("Test Size (%)", 10, 50, 20, key="clf_test_size") / 100
            
            if st.button("Run Classification Model"):
                if target_col and feature_cols:
                    try:
                        X = df[feature_cols]
                        y = df[target_col]
                        
                        # Handle categorical targets like strings
                        if y.dtype == 'object' or y.dtype.name == 'category':
                            y = y.astype('category').cat.codes
                            
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                        
                        # Scaling for distance-based algorithms
                        if clf_type in ["Logistic Regression", "K-Nearest Neighbors (KNN)", "SVM"]:
                            scaler = StandardScaler()
                            X_train = scaler.fit_transform(X_train)
                            X_test = scaler.transform(X_test)
                            
                        # Hyperparameter Tuning
                        with st.expander("⚙️ Model Parameters (Advanced)", expanded=False):
                            params = {}
                            if clf_type == "Logistic Regression":
                                params['C'] = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
                            elif clf_type == "K-Nearest Neighbors (KNN)":
                                params['n_neighbors'] = st.slider("Neighbors (K)", 1, 20, 5)
                            elif clf_type == "Decision Tree":
                                params['max_depth'] = st.slider("Max Depth", 1, 50, 10)
                            elif clf_type == "Random Forest":
                                params['n_estimators'] = st.slider("Trees", 10, 500, 100)
                                params['max_depth'] = st.slider("Max Depth", 1, 50, 10)
                            elif clf_type == "SVM":
                                params['C'] = st.slider("C", 0.01, 10.0, 1.0)
                                params['kernel'] = st.selectbox("Kernel", ["rbf", "linear", "poly"])
                            elif clf_type == "XGBoost":
                                params['n_estimators'] = st.slider("Num Boosters", 10, 500, 100)
                                params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1)

                        # PCA option
                        use_pca = st.checkbox("🔮 Apply PCA (Dimensionality Reduction)", value=False, key="clf_pca")
                        if use_pca:
                            n_comp = st.number_input("PCA Components", 1, len(feature_cols), min(2, len(feature_cols)))
                            pca = PCA(n_components=n_comp)
                            X_train = pca.fit_transform(X_train)
                            X_test = pca.transform(X_test)

                        if clf_type == "Logistic Regression":
                            model = LogisticRegression(max_iter=1000, **params)
                        elif clf_type == "K-Nearest Neighbors (KNN)":
                            model = KNeighborsClassifier(**params)
                        elif clf_type == "Decision Tree":
                            model = DecisionTreeClassifier(**params)
                        elif clf_type == "Random Forest":
                            model = RandomForestClassifier(**params, random_state=42)
                        elif clf_type == "SVM":
                            model = SVC(**params, probability=True)
                        elif clf_type == "XGBoost":
                            model = XGBClassifier(**params, random_state=42)
                            
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        
                        # Metrics Dashboard
                        acc = accuracy_score(y_test, predictions)
                        prec = precision_score(y_test, predictions, average='weighted', zero_division=0)
                        rec = recall_score(y_test, predictions, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
                        
                        st.divider()
                        st.subheader("🏆 Classification Metrics Dashboard")
                        metrics_cols = st.columns(4)
                        metrics_cols[0].metric("Accuracy", f"{acc:.4f}")
                        metrics_cols[1].metric("Precision", f"{prec:.4f}")
                        metrics_cols[2].metric("Recall", f"{rec:.4f}")
                        metrics_cols[3].metric("F1-Score", f"{f1:.4f}")
                        
                        st.markdown("### 📋 Confusion Matrix & Class Report")
                        col_cm, col_rep = st.columns([1, 1])
                        
                        with col_cm:
                            cm = confusion_matrix(y_test, predictions)
                            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        with col_rep:
                            st.markdown("**Detailed Classification Report:**")
                            st.code(classification_report(y_test, predictions))
                            
                        st.divider()
                        st.markdown("### 💾 Export Trained Model")
                        model_bytes = pickle.dumps(model)
                        st.download_button(
                            label="📥 Download Trained Model (.pkl)",
                            data=model_bytes,
                            file_name=f"{clf_type.replace(' ', '_').lower()}.pkl",
                            mime="application/octet-stream",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"Modeling Error: {e}")
                else:
                    st.warning("Please select target and features.")
                    
        with tab3:
            st.subheader("Clustering (K-Means)")
            cluster_cols = st.multiselect("Select Variables for Clustering", numeric_cols, key="km_features")
            
            if st.button("Generate Elbow Plot") and len(cluster_cols) >= 2:
                X = df[cluster_cols]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                inertia = []
                K = range(1, 11)
                for k in K:
                    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                    kmeans.fit(X_scaled)
                    inertia.append(kmeans.inertia_)
                    
                fig = px.line(x=list(K), y=inertia, markers=True, title="Elbow Plot (Optimal K)", labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'})
                st.plotly_chart(fig, use_container_width=True)
                
            n_clusters = st.number_input("Number of Clusters (k)", min_value=2, max_value=10, value=3)
            if st.button("Run K-Means Clustering"):
                if len(cluster_cols) >= 2:
                    X = df[cluster_cols]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                    clusters = kmeans.fit_predict(X_scaled)
                    
                    df_clusters = df.copy()
                    df_clusters['Cluster'] = clusters.astype(str)
                    
                    # 2D or 3D view
                    if len(cluster_cols) == 2:
                        fig = px.scatter(df_clusters, x=cluster_cols[0], y=cluster_cols[1], color='Cluster', title="K-Means Clusters")
                        st.plotly_chart(fig, use_container_width=True)
                    elif len(cluster_cols) >= 3:
                        fig = px.scatter_3d(df_clusters, x=cluster_cols[0], y=cluster_cols[1], z=cluster_cols[2], color='Cluster', title="K-Means Clusters (First 3 Features)")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    st.session_state.data = df_clusters
                    st.success("Added 'Cluster' column to your dataset!")
                else:
                    st.warning("Please select at least 2 numerical features for clustering.")

    else:
        st.warning("Please upload data first in the Data Import module.")
