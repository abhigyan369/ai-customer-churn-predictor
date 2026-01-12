import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from utils import generate_dummy_data, load_data, preprocess_data
from model import train_model, predict_churn, get_shap_values

st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("Customer Churn Prediction Dashboard")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'explainer' not in st.session_state:
    st.session_state.explainer = None
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None

# Sidebar
with st.sidebar:
    st.header("Data & Training")
    
    data_source = st.radio("Select Data Source", ["Upload CSV", "Use Dummy Data"])
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your customer data", type="csv")
        if uploaded_file is not None:
            if st.button("Load Data"):
                st.session_state.data = load_data(uploaded_file)
                st.success("Data Loaded Successfully!")
    else:
        if st.button("Generate Dummy Data"):
            st.session_state.data = generate_dummy_data()
            st.success("Dummy Data Generated!")

    if st.session_state.data is not None:
        st.write("Data Preview:")
        st.dataframe(st.session_state.data.head())
        
        if st.button("Train Model"):
            with st.spinner("Preprocessing and Training..."):
                X, y = preprocess_data(st.session_state.data)
                
                if y is None:
                    st.error("Target column 'churn' not found for training.")
                else:
                    model, metrics, X_test, y_test = train_model(X, y)
                    explainer, shap_values = get_shap_values(model, X_test)
                    
                    st.session_state.model = model
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.explainer = explainer
                    st.session_state.shap_values = shap_values
                    
                    st.success("Model Trained!")
                    st.metric("Accuracy", f"{metrics['accuracy']:.2f}")
                    st.metric("ROC AUC", f"{metrics['roc_auc']:.2f}")

# Main Area
if st.session_state.model is not None:
    tab1, tab2, tab3 = st.tabs(["Churn Risk Analysis", "Explainability", "What-If Simulator"])
    
    with tab1:
        st.subheader("Customer Churn Risk")
        
        # Get predictions for test set
        probs = predict_churn(st.session_state.model, st.session_state.X_test)
        results = st.session_state.X_test.copy()
        results['Churn Probability'] = probs
        results['Actual Churn'] = st.session_state.y_test.values
        
        # Sort by high risk
        results = results.sort_values(by='Churn Probability', ascending=False)
        
        st.dataframe(results.style.format({'Churn Probability': '{:.2%}'}))
    
    with tab2:
        st.subheader("Model Explainability (SHAP)")
        
        st.write("Feature Importance Summary")
        fig, ax = plt.subplots()
        shap.summary_plot(st.session_state.shap_values, st.session_state.X_test, show=False)
        st.pyplot(fig)
        
        st.write("Individual Explanation")
        # Select customer
        customer_idx = st.selectbox("Select a customer index from test set", st.session_state.X_test.index)
        
        # Find integer location for SHAP
        iloc_idx = st.session_state.X_test.index.get_loc(customer_idx)
        
        st.write(f"Explanation for Customer {customer_idx}")
        st.write(f"Churn Probability: {probs[iloc_idx]:.2%}")
        
        fig_force = shap.force_plot(
            st.session_state.explainer.expected_value,
            st.session_state.shap_values[iloc_idx,:],
            st.session_state.X_test.iloc[iloc_idx,:],
            matplotlib=True,
            show=False
        )
        st.pyplot(fig_force)

    with tab3:
        st.subheader("What-If Simulator")
        
        # Select base customer
        sim_idx = st.selectbox("Select base customer profile", st.session_state.X_test.index, key='sim_select')
        base_features = st.session_state.X_test.loc[sim_idx].copy()
        
        col1, col2 = st.columns(2)
        
        # Create input widgets for top numeric features
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        adjusted_features = base_features.copy()
        
        with col1:
             for col in numeric_cols:
                 if col in base_features:
                     val = st.slider(f"Adjust {col}", 
                                     min_value=float(st.session_state.X_test[col].min()), 
                                     max_value=float(st.session_state.X_test[col].max()), 
                                     value=float(base_features[col]))
                     adjusted_features[col] = val
        
        # Recalculate probability
        # Reshape for prediction
        input_df = pd.DataFrame([adjusted_features])
        new_prob = st.session_state.model.predict_proba(input_df)[0, 1]
        
        with col2:
            st.metric("New Churn Probability", f"{new_prob:.2%}", delta=f"{new_prob - results.loc[sim_idx, 'Churn Probability']:.2%}")
            
            st.write("Comparison:")
            comp_data = {
                'Original': results.loc[sim_idx, 'Churn Probability'],
                'Simulated': new_prob
            }
            st.bar_chart(comp_data)

else:
    st.info("Please generate data and train the model using the sidebar to view insights.")
