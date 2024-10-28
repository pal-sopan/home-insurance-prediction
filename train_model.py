# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import lightgbm as lgb  # Importing LightGBM
import joblib

def train_model():
    # Load the dataset
    file_path = 'home_data.csv'  # Update this path if necessary
    df = pd.read_csv(file_path)

    # Define the target variable and features
    target_column = 'Premium'
    feature_columns = [
        'Sqfeet', 'Bedrooms', 'Bathrooms', 'Year Built', 
        'Credit Rating', 'Occupancy Status', 'Home Type', 
        'Zip', 'City', 'State', 'Gender', 
        'Are You Married'  # Removed Age from here; will calculate it later
    ]

    # Check if feature columns exist in the DataFrame
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in the DataFrame: {missing_features}")

    # Calculate Age from DOB and add it to features (assuming DOB is in datetime format)
    if 'DOB' not in df.columns:
        raise ValueError("DOB column is missing from the DataFrame.")
    
    df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')  # Convert to datetime, handle errors if any
    df['Age'] = (pd.Timestamp.now() - df['DOB']).dt.days // 365  # Calculate age in years

    # Add Age to feature columns for X
    feature_columns.append('Age')

    # Define features (X) and target variable (y)
    X = df[feature_columns]  # Using only the selected feature columns
    y = df[target_column]     # Target column 'Premium'

    # Identify categorical and numerical columns for preprocessing
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Build pipelines for preprocessing
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
        ('scaler', StandardScaler())                   # Standardize features
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with most frequent value
        ('encoder', OneHotEncoder(handle_unknown='ignore'))     # One-hot encode categorical variables
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the preprocessor on training data and transform both train and test sets
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # LightGBM Model without Hyperparameter Tuning
    lgb_model = lgb.LGBMRegressor(random_state=42)

    # Fit the model on the training data
    lgb_model.fit(X_train_preprocessed, y_train)

    # Save the trained model as a .pkl file using joblib 
    joblib.dump((lgb_model, preprocessor), 'home_model.pkl')  # Save both model and preprocessor
    
    print("Model trained and saved as home_model.pkl")

# Function to predict the premium using the trained model
def predict_premium(input_data):
    # Load the trained model and preprocessor
    model, preprocessor = joblib.load('home_model.pkl')

    # Preprocess the input data
    input_preprocessed = preprocessor.transform(input_data)

    # Make prediction
    prediction = model.predict(input_preprocessed)

    return prediction[0]  # Return the first prediction

# Call the function to train the model (this should be done only once)
if __name__ == "__main__":
    train_model()
