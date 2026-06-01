import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load Pima Indians Diabetes Dataset
# Dataset from Kaggle: https://www.kaggle.com/uciml/pima-indians-diabetes-database
# Using direct URL for the CSV
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
try:
    df = pd.read_csv(url, names=column_names)
    print("Dataset loaded successfully from URL.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Preprocessing
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Handle missing values (replace 0 with mean for relevant columns)
X['Glucose'] = X['Glucose'].replace(0, X['Glucose'].mean())
X['BloodPressure'] = X['BloodPressure'].replace(0, X['BloodPressure'].mean())
X['SkinThickness'] = X['SkinThickness'].replace(0, X['SkinThickness'].mean())
X['Insulin'] = X['Insulin'].replace(0, X['Insulin'].mean())
X['BMI'] = X['BMI'].replace(0, X['BMI'].mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Save model
joblib.dump(model, 'diabetes_model.pkl')
print('Model saved as diabetes_model.pkl')
