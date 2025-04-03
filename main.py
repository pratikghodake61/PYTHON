import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from flask import Flask, request, jsonify

# Load the dataset
data = pd.read_csv('weather_data.csv')

# Display the first few rows of the dataset
print("Initial DataFrame:")
print(data.head())

# Display the columns to check for any issues
print("Columns in DataFrame:")
print(data.columns)

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# Fill missing values using forward fill
data.ffill(inplace=True)  # Forward fill for simplicity

# Convert date to datetime format
data['date'] = pd.to_datetime(data['date'])

# Set the date as the index
data.set_index('date', inplace=True)

# Display the cleaned data
print("Cleaned DataFrame:")
print(data.head())

# Check if the DataFrame is empty
if data.empty:
    print("The DataFrame is empty. Please check the CSV file.")
else:
    # Plot temperature over time
    plt.figure(figsize=(12, 6))
    plt.plot(data['meantemp'], label='Mean Temperature', color='orange')
    plt.title('Mean Temperature Over Time')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.show()

    # Correlation matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Features and target variable
    try:
        X = data[['humidity', 'wind_speed', 'meanpressure']]
        y = data['meantemp']  # Assuming we want to predict mean temperature

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f'Mean Squared Error: {mse}')
        print(f'R^2 Score: {r2}')

        # Hyperparameter tuning using GridSearchCV
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

        # Fit the model
        grid_search.fit(X_train, y_train)

        # Best parameters
        print(f'Best Parameters: {grid_search.best_params_}')

    except KeyError as e:
        print(f"KeyError: {e}. Please check the column names in the DataFrame.")