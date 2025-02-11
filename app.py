import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from flask import Flask, request, jsonify
import joblib

# Load the dataset
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv('D:/WebiSoftTech/various Machine Learning Models/processed.cleveland.data', names=column_names, na_values='?')

# Handle missing values by dropping rows with NaN values
data.dropna(inplace=True)

# Features and target
X = data.drop('target', axis=1)
y = np.where(data['target'] > 0, 1, 0)  # Binary classification: 0 for no disease, 1 for disease

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models and hyperparameters
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC()
}

params = {
    "Logistic Regression": {"C": [0.1, 1, 10]},
    "Random Forest": {"n_estimators": [50, 100, 200]},
    "Support Vector Machine": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
}

best_model = None
best_accuracy = 0
best_model_name = ""

# Train and evaluate models
for name, model in models.items():
    grid_search = GridSearchCV(model, params[name], cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred = grid_search.best_estimator_.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = grid_search.best_estimator_
        best_model_name = name

print(f"Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# Save the best model and scaler
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return "Heart Disease Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array([data[feature] for feature in X.columns]).reshape(1, -1)
        
        # Load model and scaler
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        
        # Scale input features
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
        
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)