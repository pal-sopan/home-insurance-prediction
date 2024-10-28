from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import logging
import os
import mysql.connector  # Import MySQL connector

# Import necessary libraries for model preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import lightgbm as lgb  # Importing LightGBM

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the pre-trained model and preprocessor
try:
    model, preprocessor = joblib.load('home_model.pkl')  # Ensure the correct model and preprocessor are loaded
except Exception as e:
    logger.error(f"Error loading the model and preprocessor: {str(e)}")
    model = None
    preprocessor = None

# MySQL database connection
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',  # Your MySQL host
        user='root',  # Your MySQL username
        password='admin',  # Your MySQL password
        database='home_insurance'  # Your database name
    )

def upload_csv_to_db(csv_file_path):
    conn = None
    cursor = None
    try:
        df = pd.read_csv(csv_file_path)
        print(df.columns)  # Print the columns to check them
        
        # Check for required columns
        required_columns = ['gender', 'age', 'sqfeet', 'bedrooms', 'bathrooms', 
                            'year_built', 'credit_rating', 'occupancy_status', 
                            'home_type', 'zip', 'city', 'state', 'are_you_married']
        
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f'Missing column in CSV: {col}')
            
        df['are_you_married'] = df['are_you_married'].map({'Yes': 1, 'No': 0})    

        conn = get_db_connection()
        cursor = conn.cursor()

        for index, row in df.iterrows():
            cursor.execute(
                """
                INSERT INTO users (gender, age, sqfeet, bedrooms, bathrooms, year_built, credit_rating, occupancy_status, home_type, zip, city, state, are_you_married)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (row['gender'], row['age'], row['sqfeet'], row['bedrooms'],
                 row['bathrooms'], row['year_built'], row['credit_rating'],
                 row['occupancy_status'], row['home_type'], row['zip'],
                 row['city'], row['state'], row['are_you_married'])
            )

        conn.commit()
        print("CSV data uploaded to MySQL database successfully.")

    except Exception as e:
        print(f'Error uploading CSV data to MySQL: {str(e)}')

    finally:
        if cursor is not None:
            cursor.close()  # Close cursor only if it was created
        if conn is not None:
            conn.close()  # Close connection only if it was created

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')  # Adjust path as needed

@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    csv_file_path = r'C:\Users\HP\Desktop\Home insurance prediction price\submitted_data.csv'
    upload_csv_to_db(csv_file_path)
    return "CSV uploaded successfully!"

# Route to update the model with new data
@app.route('/update-model', methods=['POST'])
def update_model():
    """Update the model with new training data."""
    from train_model import train_model  # Import the function to retrain the model
    try:
        train_model()  # Call the training function
        global model, preprocessor
        model, preprocessor = joblib.load('home_model.pkl')  # Reload updated model and preprocessor
        return jsonify({'message': 'Model updated successfully!'}), 200
    except Exception as e:
        logger.error(f'Error updating model: {str(e)}')
        return jsonify({'error': 'Failed to update model'}), 500

# Mapping of input data keys to expected model features
column_mapping = {
    'gender': 'Gender',
    'age': 'Age',
    'sqfeet': 'Sqfeet',
    'bedrooms': 'Bedrooms',
    'bathrooms': 'Bathrooms',
    'year_built': 'Year Built',
    'credit_rating': 'Credit Rating',
    'occupancy_status': 'Occupancy Status',
    'home_type': 'Home Type',
    'zip': 'Zip',
    'city': 'City',
    'state': 'State',
    'are_you_married': 'Are You Married'
}

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction based on input data and store submitted data in MySQL database."""
    try:
        # Parse JSON data from request
        data = request.get_json()

        # Log received data for debugging
        logger.info(f"Received data: {data}")

        # Validate incoming data
        if not data or not all(key in data for key in column_mapping.keys()):
            return jsonify({'error': 'Invalid input data'}), 400

        # Map incoming data to expected columns for the model
        model_data = {column_mapping[key]: value for key, value in data.items() if key in column_mapping}
        model_data['Are You Married'] = 1 if model_data['Are You Married'] == 'Yes' else 0

        # Convert to DataFrame for prediction
        df = pd.DataFrame([model_data])

        # Check for missing columns that the model expects
        expected_columns = set(column_mapping.values())
        missing_columns = expected_columns - set(df.columns)
        if missing_columns:
            return jsonify({'error': f'Columns are missing: {missing_columns}'}), 400

        # Preprocess the data using the saved preprocessor
        df_preprocessed = preprocessor.transform(df)

        # Log preprocessed data for debugging
        logger.info(f"Preprocessed data: {df_preprocessed}")

        # Make prediction using the pre-trained model
        prediction = model.predict(df_preprocessed)
        predicted_premium = prediction[0]  # Assuming model returns a single value

        # Log prediction result for debugging
        logger.info(f"Prediction: {predicted_premium}")

        # Store the submitted data in the MySQL database
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Insert user data into the users table
            cursor.execute(
                """
                INSERT INTO users (gender, age, sqfeet, bedrooms, bathrooms, year_built, credit_rating, occupancy_status, home_type, zip, city, state, are_you_married)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, 
                (model_data['Gender'], model_data['Age'], model_data['Sqfeet'], model_data['Bedrooms'],
                 model_data['Bathrooms'], model_data['Year Built'], model_data['Credit Rating'],
                 model_data['Occupancy Status'], model_data['Home Type'], model_data['Zip'],
                 model_data['City'], model_data['State'], model_data['Are You Married'])
            )
            user_id = cursor.lastrowid  # Get the last inserted user ID

            # Insert prediction into the predictions table
            cursor.execute(
                """
                INSERT INTO predictions (user_id, predicted_premium)
                VALUES (%s, %s)
                """, 
                (user_id, predicted_premium)
            )
            conn.commit()
            logger.info("Data saved to MySQL database successfully.")
        except Exception as db_error:
            logger.error(f"Error saving data to MySQL: {str(db_error)}")
        finally:
            cursor.close()
            conn.close()
            
        # Store the submitted data in a CSV file
        csv_file_path = 'submitted_data.csv'
        try:
            if not os.path.isfile(csv_file_path):
                df.to_csv(csv_file_path, mode='w', header=True, index=False)  # Create file and write header
            else:
                df.to_csv(csv_file_path, mode='a', header=False, index=False)  # Append to existing file
            logger.info(f"Data saved to {csv_file_path} successfully.")
        except Exception as file_error:
            logger.error(f"Error saving data to CSV: {str(file_error)}")

        # Call the training function to retrain the model
        from train_model import train_model  # Import here to avoid circular import issues
        train_model()  # Assuming this function handles the retraining and saving of the model

        return jsonify({'predicted_premium': round(predicted_premium, 2)}), 200

    except Exception as e:
        logger.error(f'Error during prediction: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    if model is None or preprocessor is None:
        logger.error("Model or preprocessor is not loaded, exiting.")
    else:
        app.run(debug=True)
