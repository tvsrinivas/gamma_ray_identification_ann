import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')


with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.title('Gamma Ray Identification')

# User input
fLength = st.number_input('fLength')
fWidth = st.number_input('fWidth')
fSize = st.number_input('fSize')
fConc = st.number_input('fConc')
fConc1 = st.number_input('fConc1')
fAsym = st.number_input('fAsym')
fM3Long = st.number_input('fM3Long')
fM3Trans = st.number_input('fM3Trans')
fAlpha = st.number_input('fAlpha')
fDist = st.number_input('fDist')

# Prepare the input data
input_data = pd.DataFrame({
    'fLength': [fLength],
    'fWidth': [fWidth],
    'fSize': [fSize],
    'fConc': [fConc],
    'fConc1': [fConc1],
    'fAsym': [fAsym],
    'fM3Long': [fM3Long],
    'fM3Trans': [fM3Trans],
    'fAlpha': [fAlpha],
    'fDist': [fDist]
})


# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Gammaa Ray Probability: {prediction_proba:.2f}')



if prediction_proba < 0.5 :
    st.write(" It is a Gamma Signal")
else :
    st.write(" It is a hadron(Background signal) ")
