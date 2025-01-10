import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
# from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load the trained model
model = tf.keras.models.load_model('churn_model.h5')

## load the encoder and scalar
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geography.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('scalar.pkl', 'rb') as file:
    scaler = pickle.load(file)

## streamlit app
st.title('Customer Churn Prediction')

# User Input
geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.number_input('Age', min_value=18, max_value=100)
balance = st.number_input('Balance', min_value=0)
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.number_input('Tenure', 0, 10)
num_of_products = st.number_input('Number of Products', 1, 4)
has_credit_card = st.number_input('Has Credit Card', [0, 1])
is_active_member = st.number_input('Is Active Member', [0, 1])

## Prepare the data
input_data = pd.DataFrame({
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'CreditScore': [credit_score],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

## Encode geography
geo_encoded = label_encoder_geo.transform([[geography]]).toarray() # converted into array
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography'])) # converted back into data frame

# Combine one hot encoded geography with the input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1) 
## Error was coming -> The feature names should match those that were passed during fit. Feature names must be in the same order as they were in fit.

## Scale the data
input_data_scaled = scaler.transform(input_data)

## Make the prediction
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

## Display the prediction
if prediction_prob > 0.5:
    st.write(f'Customer is likely to churn with a probability of {prediction_prob:.2f}')
else:
    st.write(f'Customer is likely to remain with the bank with a probability of {1 - prediction_prob:.2f}')