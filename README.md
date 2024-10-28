# Home Insurance Premium Predictor

This project is a Flask-based web application that predicts home insurance premiums based on various input parameters. It uses a machine learning model trained on historical data to make predictions.

## Features

- Predicts home insurance premiums based on user input
- Updates the model with new data
- Stores submitted data in a CSV file
- Uses LightGBM for machine learning predictions
- Provides a web interface for easy interaction

## Prerequisites

- Python 3.7+
- MySQL database

## Install Dependencies

- pip install -r requirements.txt


Ensure you have the trained model file `home_model.pkl` in the project directory.

## Configuration

Update the following configurations in the `app.py` file:
- Database connection string
- Any other environment-specific settings