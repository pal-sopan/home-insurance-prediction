# # Import necessary libraries
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV, KFold
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import lightgbm as lgb  # Importing LightGBM

# # Load the dataset
# file_path = 'home_data.csv'  # Update this path if necessary
# df = pd.read_csv(file_path)

# # Check for missing values and basic stats
# print(df.info())
# print(df.describe())

# # Define the target variable and features
# target_column = 'Premium'  # Define the target column

# # Adjust feature columns based on available columns in the DataFrame
# feature_columns = [
#     'Sqfeet',            # Square footage of the home
#     'Bedrooms',          # Number of bedrooms
#     'Bathrooms',         # Number of bathrooms
#     'Year Built',        # Year the home was built
#     'Credit Rating',     # Credit rating of the homeowner
#     'Occupancy Status',  # Occupancy status (if applicable)
#     'Home Type',         # Type of home (e.g., Single Family, Duplex)
#     'Zip',               # Zip code of the property
#     'City',              # City where the property is located
#     'State',             # State where the property is located
#     'Gender',            # Gender of the homeowner
#     'DOB'                # Date of Birth for calculating Age later
# ]

# # Check if feature columns exist in the DataFrame
# missing_features = [col for col in feature_columns if col not in df.columns]
# if missing_features:
#     raise ValueError(f"Missing features in the DataFrame: {missing_features}")

# # Define features (X) and target variable (y)
# X = df[feature_columns]  # Using only the selected feature columns
# y = df[target_column]     # Target column 'Premium'

# # Calculate Age from DOB and add it to features (assuming DOB is in datetime format)
# df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')  # Convert to datetime, handle errors if any
# df['Age'] = (pd.Timestamp.now() - df['DOB']).dt.days // 365  # Calculate age in years

# # Add Age to feature columns for X
# feature_columns.append('Age')

# # Define features (X) again after adding Age
# X = df[feature_columns]

# # Identify categorical and numerical columns for preprocessing
# categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
# numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# # Build pipelines for preprocessing

# # Numerical preprocessing: Impute missing values and scale features
# numerical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
#     ('scaler', StandardScaler())                   # Standardize features
# ])

# # Categorical preprocessing: Impute missing values and encode categorical variables
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with most frequent value
#     ('encoder', OneHotEncoder(handle_unknown='ignore'))     # One-hot encode categorical variables
# ])

# # Combine numerical and categorical transformers into a single preprocessor
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numerical_transformer, numerical_cols),
#         ('cat', categorical_transformer, categorical_cols)
#     ])

# # Split the data into training and testing sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Fit the preprocessor on training data and transform both train and test sets
# X_train_preprocessed = preprocessor.fit_transform(X_train)
# X_test_preprocessed = preprocessor.transform(X_test)

# # LightGBM Model with Hyperparameter Tuning
# lgb_model = lgb.LGBMRegressor(random_state=42)
# lgb_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('model', lgb_model)
# ])

# # Define hyperparameter grid for tuning
# param_grid_lgb = {
#     'model__n_estimators': [100, 200],  # Consider adding more options for tuning
#     'model__max_depth': [-1, 10, 20],   # Consider adding more options for depth
# }

# # Use GridSearchCV for hyperparameter tuning with LightGBM
# lgb_search = GridSearchCV(lgb_pipeline, param_grid_lgb,
#                            cv=KFold(n_splits=5, shuffle=True, random_state=42),
#                            scoring='neg_mean_squared_error')

# # Fit the model on the training data
# lgb_search.fit(X_train, y_train)

# # Get the best estimator from GridSearchCV
# best_lgb = lgb_search.best_estimator_

# # Evaluate the best LightGBM model
# def evaluate_model(model, X_test, y_test):
#     """Evaluate model performance using RMSE, MAE, and R²."""
#     predictions = model.predict(X_test)
#     rmse = np.sqrt(mean_squared_error(y_test, predictions))
#     mae = mean_absolute_error(y_test, predictions)
#     r2 = r2_score(y_test, predictions)
#     return rmse, mae, r2

# lgb_rmse, lgb_mae, lgb_r2 = evaluate_model(best_lgb.named_steps['model'], X_test_preprocessed, y_test)

# # Display results for LightGBM
# print(f"LightGBM RMSE: {lgb_rmse:.2f}")
# print(f"LightGBM MAE: {lgb_mae:.2f}")
# print(f"LightGBM R²: {lgb_r2:.2f}")

# # Output the best model based on RMSE performance metric
# print(f"Best LightGBM model with RMSE of {lgb_rmse:.2f}")

import sys
print("syspath: ", sys.path)
