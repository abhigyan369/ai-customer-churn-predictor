import pandas as pd
import numpy as np
from io import StringIO

def generate_dummy_data(n_samples=1000):
    """Generates a synthetic Telco Customer Churn dataset."""
    np.random.seed(42)
    
    data = {
        'customerID': [f'CUST-{i:04d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(1, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n_samples),
        'MonthlyCharges': np.random.uniform(18.25, 118.75, n_samples).round(2),
        'TotalCharges': np.random.uniform(18.25, 8000.0, n_samples).round(2), # Simplified
        'churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])
    }
    
    df = pd.DataFrame(data)
    # Ensure TotalCharges is somewhat correlated with tenure * MonthlyCharges
    df['TotalCharges'] = (df['tenure'] * df['MonthlyCharges'] * np.random.uniform(0.9, 1.1, n_samples)).round(2)
    return df

def load_data(file_buffer):
    """Loads data from a file buffer (uploaded file)."""
    try:
        df = pd.read_csv(file_buffer)
        return df
    except Exception as e:
        return None

def preprocess_data(df):
    """
    Preprocesses the dataframe for modeling.
    """
    df = df.copy()
    
    # NEW: Standardize column names to lowercase to avoid case-sensitivity issues
    df.columns = [c.lower() for c in df.columns]
    
    # Drop ID if exists (changed to lowercase)
    if 'customerid' in df.columns:
        df = df.drop(columns=['customerid'])
        
    # Convert TotalCharges to numeric (changed to lowercase)
    if 'totalcharges' in df.columns:
        df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
        df['totalcharges'] = df['totalcharges'].fillna(df['totalcharges'].median())

    # Map target variable (now looking for 'churn' lowercase)
    y = None
    if 'churn' in df.columns:
        if df['churn'].dtype == 'object':
             df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})
        y = df['churn']
        df = df.drop(columns=['churn'])
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # One-hot encoding
    X = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return X, y