# Makaan.com-Rental-Price-Prediction-Model
Mumbai Rental Flat Price Prediction
This project aims to predict the rental prices of flats in Mumbai using data scraped from Makaan.com. The final model, selected for its high accuracy, is implemented using the XGBoost algorithm. Additionally, a web application has been developed using Flask to make these predictions easily accessible.

Project Structure
The repository contains the following main components:

Jupyter Notebooks

webscraping.ipynb: Contains the code to scrape rental flat data from Makaan.com.
modeling.ipynb: Includes the preprocessing steps and the implementation of various machine learning algorithms, ultimately selecting XGBoost for its performance.
Web Application

app.py: The main Flask application file.
encoder.pkl: Encoder file used in the preprocessing steps.
xgb_model.pkl: Trained XGBoost model used for predictions.
templates/index.html: HTML template for the web application's front end.
Getting Started
Prerequisites
Ensure you have the following installed:

Python 3.6 or higher
Jupyter Notebook
Flask
Required Python packages listed in requirements.txt

Project Overview
Data Collection
The data is scraped from Makaan.com, focusing on rental flats in Mumbai. The webscraping.ipynb notebook contains all the necessary code for this process.

Data Preprocessing and Modeling
The modeling.ipynb notebook covers the preprocessing steps, such as handling missing values, encoding categorical variables, and scaling. Multiple machine learning algorithms were evaluated, with XGBoost selected for its superior performance.

Web Application
The web application, developed using Flask, provides an easy-to-use interface for predicting rental prices. Users can input features related to a flat, and the application will return the predicted rental price using the trained XGBoost model.

Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgements
Data source: Makaan.com
Machine Learning Library: XGBoost
