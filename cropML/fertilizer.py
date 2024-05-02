from flask import Blueprint, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

fertilizer_suggestion = Blueprint('fertilizer_suggestion', __name__)

# Load the dataset
data = pd.read_csv("fertilizer_recommendation.csv")

# Label encoding for categorical features
le_soil = LabelEncoder()
data['Soil Type'] = le_soil.fit_transform(data['Soil Type'])
le_crop = LabelEncoder()
data['Crop Type'] = le_crop.fit_transform(data['Crop Type'])

# Get the encoded soil classes
soil_classes_encoded = le_soil.classes_

# Splitting the data into input and output variables
X = data.iloc[:, :8]
y = data.iloc[:, -1]

# Training the Decision Tree Classifier model
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X, y)

@fertilizer_suggestion.route('/fertilizer_suggestion', methods=['GET', 'POST'])
def fertilizer():
    N, P, K, temperature, humidity, soil_moisture, soil_type, crop_type = '', '', '', '', '', '', '', ''
    fertilizer = ''

    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        soil_moisture = float(request.form['soil_moisture'])
        soil_type = request.form['soil_type']
        crop_type = request.form['crop_type']

        # Convert categorical values to numerical using mapping
        soil_enc = le_soil.transform([soil_type])[0]
        crop_enc = le_crop.transform([crop_type])[0]

        # Check for unseen labels
        if soil_type not in soil_classes_encoded:
            return "Unseen soil label: {}".format(soil_type)
        
        if crop_type not in le_crop.classes_:
            return "Unseen crop label: {}".format(crop_type)

        # Get the user inputs and store them in a numpy array
        user_input = [[N, P, K, temperature, humidity, soil_moisture, soil_enc, crop_enc]]

        # Predict the fertilizer name
        fertilizer_name = dtc.predict(user_input)

        # Return the prediction as a string
        fertilizer = str(fertilizer_name[0])

    return render_template('fertilizer.html', N=N, P=P, K=K, temperature=temperature, humidity=humidity,
                           soil_moisture=soil_moisture, soil_type=soil_type, crop_type=crop_type,
                           fertilizer=fertilizer)
