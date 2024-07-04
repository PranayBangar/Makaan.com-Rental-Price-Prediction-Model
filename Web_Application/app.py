from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb  # Ensure xgboost is imported

app = Flask(__name__)

# Define the tokenizer function
def amenities_tokenizer(x):
    return x.split(', ')

# Load the model
with open('xgb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the encoder
with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

# Load the vectorizer data
with open('vectorizer_data.pkl', 'rb') as vectorizer_file:
    vectorizer_data = pickle.load(vectorizer_file)

# Reconstruct the CountVectorizer
vectorizer = CountVectorizer(tokenizer=amenities_tokenizer)
vectorizer.vocabulary_ = vectorizer_data['vocabulary_']
vectorizer.stop_words_ = vectorizer_data['stop_words_']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.form.to_dict()
    square_feet = float(features['Square Feet'])
    bhk_apartment = features['BHK Apartment']
    furnish_status = features['Furnish Status']
    deposit_status = float(features['Deposit Status'])
    bathroom_count = features['Bathroom Count']
    location = features['Location']
    amenities = features['Amenities']

    amenities_vector = vectorizer.transform([amenities])
    amenities_df = pd.DataFrame(amenities_vector.toarray(), columns=vectorizer.get_feature_names_out())

    input_data = {
        'Square Feet': [square_feet],
        'BHK Apartment': [bhk_apartment],
        'Furnish Status': [furnish_status],
        'Deposit Status': [deposit_status],
        'Bathroom Count': [bathroom_count],
        'Location': [location]
    }

    input_df = pd.DataFrame(input_data)
    input_df[["BHK Apartment", "Bathroom Count", "Location", 'Furnish Status']] = encoder.transform(input_df[["BHK Apartment", "Bathroom Count", "Location", 'Furnish Status']])

    final_input_df = pd.concat([input_df, amenities_df], axis=1)

    # Ensure the columns match the model's expected input features
    model_features = model.get_booster().feature_names
    final_input_df = final_input_df.reindex(columns=model_features, fill_value=0)

    # Make predictions
    prediction = model.predict(final_input_df)

    return render_template(
        'index.html', 
        prediction_text=f'Estimated Rent Price: ₹{prediction[0]:,.2f}',
        input_data=features
    )

if __name__ == "__main__":
    app.run(debug=True)
