# Customer Churn Prediction App- https://ai-customer-churn-predictor.streamlit.app/

An AI-powered web application to predict customer churn, visualize risks, and explain predictions using SHAP values.

## Features
- **Data Upload**: Upload your own customer data (CSV).
- **Model Training**: Train an XGBoost Classifier on the fly.
- **Churn Prediction**: View churn probabilities for each customer.
- **Explainability**: Understand why a customer is at risk with SHAP plots.
- **What-If Simulation**: Adjust customer attributes to see how it affects churn risk.

## Setup

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run src/app.py
   ```

## File Structure
- `src/app.py`: Streamlit application.
- `src/model.py`: Machine learning model logic.
- `src/utils.py`: Data loading, processing, and dummy data generation.
- `data/`: Directory for storing datasets.
