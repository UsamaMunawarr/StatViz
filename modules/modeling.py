import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import pickle
import io
import statsmodels.api as sm

def generate_model_pdf(model_name, metrics, feature_df=None, spss_table=None):
    from fpdf import FPDF
    import tempfile
    import os
    import matplotlib.pyplot as plt
    
    class ModelPDFReport(FPDF):
        def header(self):
            self.set_font("Times", "B", 16)
            self.set_text_color(0, 102, 204)
            self.cell(0, 10, "StatViz - ML & Statistical Modeling Report", align="C")
            self.ln(15)
            
    pdf = ModelPDFReport()
    pdf.add_page()
    
    pdf.set_font("Times", "B", 14)
    pdf.cell(0, 10, f"Model Executed: {model_name}")
    pdf.ln(10)
    
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 8, "1. Performance Metrics")
    pdf.ln(6)
    
    pdf.set_font("Times", "", 12)
    for k, v in metrics.items():
        pdf.cell(0, 8, f"{k}: {v:.4f}")
        pdf.ln(6)
    pdf.ln(10)
    
    if spss_table is not None:
        pdf.set_font("Times", "B", 12)
        pdf.cell(0, 8, "2. Statistical Coefficients (SPSS Style)")
        pdf.ln(6)
        
        pdf.set_font("Times", "B", 9)
        col_width = 30
        if len(spss_table.columns) > 5:
            col_width = 25
            
        pdf.cell(col_width, 8, "Variable", border=1)
        for col in spss_table.columns:
            pdf.cell(col_width, 8, str(col)[:10], border=1)
        pdf.ln(8)
        
        pdf.set_font("Times", "", 9)
        for idx in spss_table.index:
            pdf.cell(col_width, 8, str(idx)[:15], border=1)
            for col in spss_table.columns:
                val = spss_table.loc[idx, col]
                val_str = f"{val:.3f}" if isinstance(val, (int, float)) and not np.isnan(val) else ""
                if str(val) == "nan": val_str = ""
                pdf.cell(col_width, 8, val_str, border=1)
            pdf.ln(8)
        pdf.ln(10)
        
    if feature_df is not None:
        pdf.set_font("Times", "B", 12)
        pdf.cell(0, 8, "3. Feature Importance Analysis")
        pdf.ln(6)
        
        plt.figure(figsize=(7, 5))
        plt.barh(feature_df['Feature'].astype(str), feature_df['Importance'], color="#00d4aa")
        plt.xlabel("Importance Score")
        plt.title("Feature Importance")
        plt.tight_layout()
        
        temp_path = os.path.join(tempfile.gettempdir(), "feat_imp.png")
        plt.savefig(temp_path, dpi=120)
        plt.close()
        
        pdf.image(temp_path, w=150)
        os.remove(temp_path)
        
    return bytes(pdf.output())

def forward_selection(X, y, significance_level=0.05, is_classification=False):
    initial_features = X.columns.tolist()
    best_features = []
    while len(initial_features) > 0:
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features, dtype=float)
        for new_column in remaining_features:
            model_X = sm.add_constant(X[best_features + [new_column]])
            if is_classification:
                try:
                    model = sm.Logit(y, model_X).fit(disp=0)
                    new_pval[new_column] = model.pvalues[new_column]
                except:
                    new_pval[new_column] = 1.0
            else:
                model = sm.OLS(y, model_X).fit()
                new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if min_p_value < significance_level:
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features

def backward_elimination(X, y, significance_level=0.05, is_classification=False):
    features = X.columns.tolist()
    while len(features) > 0:
        features_with_constant = sm.add_constant(X[features])
        if is_classification:
            try:
                model = sm.Logit(y, features_with_constant).fit(disp=0)
                p_values = model.pvalues[1:]
            except:
                break
        else:
            p_values = sm.OLS(y, features_with_constant).fit().pvalues[1:]
            
        max_p_value = p_values.max()
        if max_p_value >= significance_level:
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break 
    return features

def generate_spss_linear_table(X, y):
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = pd.DataFrame(scaler_x.fit_transform(X), columns=X.columns)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    X_scaled_const = sm.add_constant(X_scaled)
    model_std = sm.OLS(y_scaled, X_scaled_const).fit()
    
    df_res = pd.DataFrame()
    df_res['B'] = model.params.values
    df_res['Std. Error'] = model.bse.values
    
    beta_vals = model_std.params.values.copy()
    beta_vals[0] = np.nan 
    df_res['Beta'] = beta_vals
    
    df_res['t'] = model.tvalues.values
    df_res['Sig.'] = model.pvalues.values
    
    df_res.index = ['(Constant)'] + list(X.columns)
    return df_res

def generate_spss_logistic_table(X, y):
    if len(np.unique(y)) > 2:
        return pd.DataFrame({"Error": ["SPSS tables for Logistic Regression are exclusively supported for Binary Target Variables (exactly 2 unique categories)."]})
        
    X_const = sm.add_constant(X)
    try:
        model = sm.Logit(y, X_const).fit(disp=0)
    except Exception as e:
        return pd.DataFrame({"Error": [f"Convergence failed: {e}"]})
    
    df_res = pd.DataFrame()
    df_res['B'] = model.params.values
    df_res['S.E.'] = model.bse.values
    df_res['Wald'] = (model.params.values / model.bse.values) ** 2
    df_res['df'] = 1
    df_res['Sig.'] = model.pvalues.values
    df_res['Exp(B)'] = np.exp(model.params.values)
    
    df_res.index = ['(Constant)'] + list(X.columns)
    return df_res

def get_hyperparameters(model_type, prefix=""):
    params = {}
    with st.expander("⚙️ Advanced Hyperparameter Tuning", expanded=False):
        if model_type in ["Simple Linear Regression", "Multiple Linear Regression"]:
            st.info("No advanced hyperparameters for basic Linear Regression.")
        elif model_type in ["Polynomial Regression"]:
            params['degree'] = st.slider("Polynomial Degree", 2, 5, 2, key=f"{prefix}poly")
        elif model_type == "Logistic Regression":
            params['C'] = st.slider("Regularization Inverse (C)", 0.01, 10.0, 1.0, key=f"{prefix}C")
            params['max_iter'] = st.slider("Max Iterations", 100, 2000, 1000, key=f"{prefix}iter")
        elif model_type in ["K-Nearest Neighbors (KNN)"]:
            params['n_neighbors'] = st.slider("Neighbors (K)", 1, 50, 5, key=f"{prefix}k")
            params['weights'] = st.selectbox("Weights", ["uniform", "distance"], key=f"{prefix}w")
        elif model_type in ["Decision Tree"]:
            params['max_depth'] = st.slider("Max Depth", 1, 100, 10, key=f"{prefix}md")
            params['min_samples_split'] = st.slider("Min Samples Split", 2, 20, 2, key=f"{prefix}mss")
        elif model_type in ["Random Forest"]:
            params['n_estimators'] = st.slider("Num Trees", 10, 500, 100, key=f"{prefix}ne")
            params['max_depth'] = st.slider("Max Depth", 1, 100, 10, key=f"{prefix}rfmd")
        elif model_type in ["Support Vector Regressor (SVR)", "Support Vector Classifier (SVC)"]:
            params['C'] = st.slider("Regularization Cost (C)", 0.1, 100.0, 1.0, key=f"{prefix}svcc")
            params['kernel'] = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key=f"{prefix}ker")
        elif model_type in ["XGBoost"]:
            params['n_estimators'] = st.slider("Num Boosters", 10, 500, 100, key=f"{prefix}xgbe")
            params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1, key=f"{prefix}xgblr")
        elif model_type in ["Artificial Neural Network (ANN)"]:
            st.markdown("**(Tip: Format for hidden layers is tuple, e.g. 100 or 100,50)**")
            layers_str = st.text_input("Hidden Layer Sizes", "100", key=f"{prefix}annh")
            params['hidden_layer_sizes'] = tuple([int(x) for x in layers_str.split(',') if x.strip().isdigit()]) or (100,)
            params['activation'] = st.selectbox("Activation Function", ["relu", "tanh", "logistic"], key=f"{prefix}anna")
            params['max_iter'] = st.slider("Max Iterations", 200, 2000, 500, key=f"{prefix}annm")
            params['learning_rate_init'] = st.slider("Initial Learning Rate", 0.001, 0.1, 0.001, step=0.001, format="%.3f", key=f"{prefix}annlr")
            
        use_pca = st.checkbox("🔮 Apply PCA (Dimensionality Reduction)", value=False, key=f"{prefix}pca_cb")
        n_comp = None
        if use_pca:
            n_comp = st.number_input("PCA Components", 1, 100, 2, key=f"{prefix}pca_n")
            
    return params, use_pca, n_comp

def run():
    st.header("🤖 Statistical and Machine Learning Modeling")
    if st.session_state.data is not None:
        df = st.session_state.data.dropna() 
        
        tab1, tab2, tab3 = st.tabs(["Regression", "Classification", "Clustering (K-Means)"])
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        with tab1:
            st.subheader("Regression Models")
            model_type = st.selectbox("Select Regression Model", [
                "Simple Linear Regression", "Multiple Linear Regression", "Polynomial Regression",
                "K-Nearest Neighbors (KNN)", "Decision Tree", "Random Forest",
                "Support Vector Regressor (SVR)", "XGBoost", "Artificial Neural Network (ANN)"
            ])
            
            target_col = st.selectbox("Target Variable (Y)", numeric_cols, key="reg_target")
            feature_cols = st.multiselect("Feature Variables (X)", [c for c in numeric_cols if c != target_col], key="reg_features")
            
            selection_method = "Enter (All variables)"
            if model_type in ["Multiple Linear Regression", "Logistic Regression"]:
                selection_method = st.selectbox("Variable Selection Method (SPSS-style)", ["Enter (All variables)", "Forward Selection", "Backward Elimination", "Stepwise (Forward + Backward)"], key="reg_slct")
            
            test_size = st.slider("Test Size (%)", 10, 50, 20, key="reg_test_size") / 100
            
            params, use_pca, n_comp = get_hyperparameters(model_type, prefix="reg_")
            
            if st.button("Run Regression Model"):
                if target_col and feature_cols:
                    if model_type == "Simple Linear Regression" and len(feature_cols) > 1:
                        st.error("Simple Linear Regression requires exactly 1 feature variable.")
                    else:
                        X = df[feature_cols]
                        y = df[target_col]
                        
                        if model_type == "Multiple Linear Regression" and selection_method != "Enter (All variables)":
                            if selection_method == "Forward Selection" or selection_method == "Stepwise (Forward + Backward)":
                                selected = forward_selection(X, y, is_classification=False)
                            else:
                                selected = backward_elimination(X, y, is_classification=False)
                                
                            if len(selected) == 0:
                                st.error("No variables met the significance criteria for entry/retention.")
                                st.stop()
                            st.info(f"✨ Features selected by {selection_method}: {', '.join(selected)}")
                            X = X[selected]
                            feature_cols = selected
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                        
                        spss_table = None
                        if model_type in ["Simple Linear Regression", "Multiple Linear Regression"]:
                            st.subheader("📊 Regression Coefficients (SPSS Style)")
                            spss_table = generate_spss_linear_table(X_train, y_train)
                            st.dataframe(spss_table.style.format({
                                'B': '{:.3f}', 'Std. Error': '{:.3f}', 'Beta': '{:.3f}', 
                                't': '{:.3f}', 'Sig.': '{:.3f}'
                            }, na_rep="").set_properties(subset=['B'], **{'background-color': '#fff3cd', 'color': 'black', 'font-weight': 'bold'}), use_container_width=True)
                        
                        if model_type == "Polynomial Regression":
                            poly = PolynomialFeatures(degree=params.get('degree', 2))
                            X_train = poly.fit_transform(X_train)
                            X_test = poly.transform(X_test)
                        
                        if model_type in ["K-Nearest Neighbors (KNN)", "Support Vector Regressor (SVR)", "Artificial Neural Network (ANN)"]:
                            scaler = StandardScaler()
                            X_train = scaler.fit_transform(X_train)
                            X_test = scaler.transform(X_test)
                        
                        if use_pca:
                            pca = PCA(n_components=n_comp)
                            X_train = pca.fit_transform(X_train)
                            X_test = pca.transform(X_test)

                        if model_type in ["Simple Linear Regression", "Multiple Linear Regression", "Polynomial Regression"]:
                            model = LinearRegression()
                        elif model_type == "K-Nearest Neighbors (KNN)":
                            model = KNeighborsRegressor(**params)
                        elif model_type == "Decision Tree":
                            model = DecisionTreeRegressor(**params, random_state=42)
                        elif model_type == "Random Forest":
                            model = RandomForestRegressor(**params, random_state=42)
                        elif model_type == "Support Vector Regressor (SVR)":
                            model = SVR(**params)
                        elif model_type == "XGBoost":
                            model = XGBRegressor(**params, random_state=42)
                        elif model_type == "Artificial Neural Network (ANN)":
                            model = MLPRegressor(**params, random_state=42)
                            
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        
                        st.divider()
                        st.subheader("🏆 Regression Metrics Dashboard")
                        metrics = {
                            "R² Score": r2_score(y_test, predictions),
                            "MSE": mean_squared_error(y_test, predictions),
                            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
                            "MAE": mean_absolute_error(y_test, predictions)
                        }
                        cols = st.columns(4)
                        cols[0].metric("R² Score", f"{metrics['R² Score']:.4f}")
                        cols[1].metric("MSE", f"{metrics['MSE']:.4f}")
                        cols[2].metric("RMSE", f"{metrics['RMSE']:.4f}")
                        cols[3].metric("MAE", f"{metrics['MAE']:.4f}")
                        
                        importances = None
                        feature_importance_technique = ""
                        if hasattr(model, 'feature_importances_') and not use_pca:
                            importances = model.feature_importances_
                            feature_importance_technique = "Impurity-based Feature Importance (Native to Tree Models)"
                        elif model_type in ["K-Nearest Neighbors (KNN)", "Artificial Neural Network (ANN)", "Support Vector Regressor (SVR)"] and not use_pca:
                            with st.spinner("Calculating Permutation Importance..."):
                                result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
                                importances = result.importances_mean
                                feature_importance_technique = "Permutation Feature Importance (Best for Non-Tree Models)"
                        
                        feat_df = None
                        if importances is not None:
                            st.subheader("🌟 Feature Importances")
                            st.caption(f"**Technique Applied:** {feature_importance_technique}")
                            feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values('Importance', ascending=True)
                            fig_imp = px.bar(feat_df, x="Importance", y="Feature", orientation='h', title="Feature Importance")
                            st.plotly_chart(fig_imp, use_container_width=True)
                            
                        st.markdown("### 📈 Analysis Plot")
                        fig = px.scatter(x=y_test, y=predictions, labels={'x': 'True Values', 'y': 'Predicted Values'}, title="True vs Predicted")
                        fig.add_shape(type="line", line=dict(dash='dash', color='red'), x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max())
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.divider()
                        st.markdown("### 💾 Export & Reports")
                        col_btn1, col_btn2 = st.columns(2)
                        
                        model_bytes = pickle.dumps(model)
                        col_btn1.download_button("📥 Download Trained Model (.pkl)", data=model_bytes, file_name=f"{model_type.replace(' ', '_').lower()}.pkl", mime="application/octet-stream", use_container_width=True)
                        
                        with st.spinner("Generating PDF..."):
                            pdf_bytes = generate_model_pdf(model_type, metrics, feat_df, spss_table)
                            col_btn2.download_button("📄 Download PDF Report", data=pdf_bytes, file_name=f"{model_type}_Report.pdf", mime="application/pdf", use_container_width=True)

                else:
                    st.warning("Please select target and at least one feature.")
                    
        with tab2:
            st.subheader("Classification Models")
            clf_type = st.selectbox("Select Classification Model", [
                "Logistic Regression", "K-Nearest Neighbors (KNN)", "Decision Tree", 
                "Random Forest", "Support Vector Classifier (SVC)", "XGBoost", "Artificial Neural Network (ANN)"
            ])
            
            target_col = st.selectbox("Target Class (Y)", [c for c in df.columns if df[c].nunique() < 20], key="clf_target")
            feature_cols = st.multiselect("Feature Variables (X)", [c for c in numeric_cols if c != target_col], key="clf_features")
            
            selection_method = "Enter (All variables)"
            if clf_type == "Logistic Regression":
                selection_method = st.selectbox("Variable Selection Method (SPSS-style)", ["Enter (All variables)", "Forward Selection", "Backward Elimination", "Stepwise (Forward + Backward)"], key="clf_slct")
            
            test_size = st.slider("Test Size (%)", 10, 50, 20, key="clf_test_size") / 100
            
            params, use_pca, n_comp = get_hyperparameters(clf_type, prefix="clf_")
            
            if st.button("Run Classification Model"):
                if target_col and feature_cols:
                    try:
                        X = df[feature_cols]
                        y = df[target_col]
                        
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y_enc = le.fit_transform(y)
                        y = pd.Series(y_enc, index=y.index, name=y.name)
                        
                        if clf_type == "Logistic Regression" and len(le.classes_) > 2:
                            st.warning("Logistic Regression with SPSS outputs and Stepwise selection natively requires exactly a binary (2-class) target. Multiclass modeling will be smoothly executed, but SPSS-like features will be bypassed.")
                            selection_method = "Enter (All variables)"
                            
                        if clf_type == "Logistic Regression" and selection_method != "Enter (All variables)":
                            if selection_method == "Forward Selection" or selection_method == "Stepwise (Forward + Backward)":
                                selected = forward_selection(X, y, is_classification=True)
                            else:
                                selected = backward_elimination(X, y, is_classification=True)
                                
                            if len(selected) == 0:
                                st.error("No variables met the significance criteria for entry/retention.")
                                st.stop()
                            st.info(f"✨ Features selected by {selection_method}: {', '.join(selected)}")
                            X = X[selected]
                            feature_cols = selected
                            
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                        
                        spss_table = None
                        if clf_type == "Logistic Regression":
                            st.subheader("📊 Variables in the Equation (SPSS Style)")
                            spss_table = generate_spss_logistic_table(X_train, y_train)
                            if "Error" in spss_table.columns:
                                st.error(spss_table.iloc[0, 0])
                                spss_table = None
                            else:
                                st.dataframe(spss_table.style.format({
                                    'B': '{:.3f}', 'S.E.': '{:.3f}', 'Wald': '{:.3f}', 
                                    'df': '{:.0f}', 'Sig.': '{:.3f}', 'Exp(B)': '{:.3f}'
                                }, na_rep="").set_properties(subset=['B'], **{'background-color': '#fff3cd', 'color': 'black', 'font-weight': 'bold'}), use_container_width=True)
                        
                        if clf_type in ["Logistic Regression", "K-Nearest Neighbors (KNN)", "Support Vector Classifier (SVC)", "Artificial Neural Network (ANN)"]:
                            scaler = StandardScaler()
                            X_train = scaler.fit_transform(X_train)
                            X_test = scaler.transform(X_test)
                            
                        if use_pca:
                            pca = PCA(n_components=n_comp)
                            X_train = pca.fit_transform(X_train)
                            X_test = pca.transform(X_test)

                        if clf_type == "Logistic Regression":
                            model = LogisticRegression(**params)
                        elif clf_type == "K-Nearest Neighbors (KNN)":
                            model = KNeighborsClassifier(**params)
                        elif clf_type == "Decision Tree":
                            model = DecisionTreeClassifier(**params, random_state=42)
                        elif clf_type == "Random Forest":
                            model = RandomForestClassifier(**params, random_state=42)
                        elif clf_type == "Support Vector Classifier (SVC)":
                            model = SVC(**params)
                        elif clf_type == "XGBoost":
                            model = XGBClassifier(**params, random_state=42)
                        elif clf_type == "Artificial Neural Network (ANN)":
                            model = MLPClassifier(**params, random_state=42)
                            
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        
                        st.divider()
                        st.subheader("🏆 Classification Metrics Dashboard")
                        metrics = {
                            "Accuracy": accuracy_score(y_test, predictions),
                            "Precision": precision_score(y_test, predictions, average='weighted', zero_division=0),
                            "Recall": recall_score(y_test, predictions, average='weighted', zero_division=0),
                            "F1-Score": f1_score(y_test, predictions, average='weighted', zero_division=0)
                        }
                        
                        metrics_cols = st.columns(4)
                        metrics_cols[0].metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                        metrics_cols[1].metric("Precision", f"{metrics['Precision']:.4f}")
                        metrics_cols[2].metric("Recall", f"{metrics['Recall']:.4f}")
                        metrics_cols[3].metric("F1-Score", f"{metrics['F1-Score']:.4f}")
                        
                        importances = None
                        feature_importance_technique = ""
                        if hasattr(model, 'feature_importances_') and not use_pca:
                            importances = model.feature_importances_
                            feature_importance_technique = "Impurity-based Feature Importance (Native to Tree Models)"
                        elif clf_type in ["K-Nearest Neighbors (KNN)", "Artificial Neural Network (ANN)", "Support Vector Classifier (SVC)"] and not use_pca:
                            with st.spinner("Calculating Permutation Importance..."):
                                result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
                                importances = result.importances_mean
                                feature_importance_technique = "Permutation Feature Importance (Best for Non-Tree Models)"
                        
                        feat_df = None
                        if importances is not None:
                            st.subheader("🌟 Feature Importances")
                            st.caption(f"**Technique Applied:** {feature_importance_technique}")
                            feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values('Importance', ascending=True)
                            fig_imp = px.bar(feat_df, x="Importance", y="Feature", orientation='h', title="Feature Importance")
                            st.plotly_chart(fig_imp, use_container_width=True)
                            
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
                        st.markdown("### 💾 Export & Reports")
                        col_btn1, col_btn2 = st.columns(2)
                        
                        model_bytes = pickle.dumps(model)
                        col_btn1.download_button("📥 Download Trained Model (.pkl)", data=model_bytes, file_name=f"{clf_type.replace(' ', '_').lower()}.pkl", mime="application/octet-stream", use_container_width=True)
                        
                        with st.spinner("Generating PDF..."):
                            pdf_bytes = generate_model_pdf(clf_type, metrics, feat_df, spss_table)
                            col_btn2.download_button("📄 Download PDF Report", data=pdf_bytes, file_name=f"{clf_type}_Report.pdf", mime="application/pdf", use_container_width=True)
                            
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
