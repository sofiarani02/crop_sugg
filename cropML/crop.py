from flask import Blueprint, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

crop_prediction = Blueprint('crop_prediction', __name__)

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC()
}

# Load the dataset
dataset = pd.read_csv('output.csv')

# Define feature matrix X and target vector y
X = dataset[['N', 'P', 'K', 'temperature', 'humidity', 'soil_moisture']]
y = dataset['label']

# Train and evaluate models
accuracy_results = {}
for name, model in models.items():
    # Train the model
    model.fit(X, y)
    
    # Perform cross-validation for accuracy
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    # Calculate mean accuracy
    accuracy = scores.mean()
    accuracy_results[name] = accuracy
    print(f"{name} Accuracy:", round(accuracy * 100, 2))  # Print accuracy here

# Function to find the closest row with the same crop and return its pH and rainfall values
def find_closest_crop_row(crop_label, N, P, K):
    crop_rows = dataset[dataset['label'] == crop_label]
    min_distance = float('inf')
    closest_row = None
    for index, row in crop_rows.iterrows():
        distance = abs(row['N'] - N) + abs(row['P'] - P) + abs(row['K'] - K)
        if distance < min_distance:
            min_distance = distance
            closest_row = row
    return closest_row['ph'], closest_row['rainfall']

@crop_prediction.route('/crop_prediction', methods=['GET', 'POST'])
def crop():
    N, P, K, temperature, humidity, soil_moisture = '', '', '', '', '', ''
    crop, pH, rainfall = '', '', ''

    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        soil_moisture = float(request.form['soil_moisture'])
        
        # Predict crop
        best_model_name = max(accuracy_results, key=accuracy_results.get)
        best_model = models[best_model_name]
        crop_label = best_model.predict([[N, P, K, temperature, humidity, soil_moisture]])[0]
        
        # Find optimal pH and rainfall for predicted crop
        optimal_ph, optimal_rainfall = find_closest_crop_row(crop_label, N, P, K)

        return render_template('crop.html', N=N, P=P, K=K, temperature=temperature, humidity=humidity,
                               soil_moisture=soil_moisture, crop=crop_label, pH=optimal_ph,
                               rainfall=optimal_rainfall, accuracy_results=accuracy_results)
    
    return render_template('crop.html', N=N, P=P, K=K, temperature=temperature, humidity=humidity,
                           soil_moisture=soil_moisture, crop=crop, pH=pH, rainfall=rainfall, accuracy_results=accuracy_results)
