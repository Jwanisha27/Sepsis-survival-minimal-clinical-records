import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = load_model('sepsis_survival_model.h5')

# Define a function to preprocess and predict
def predict_survival(age, gender, sepsis_episodes):
    # Create input data
    input_data = np.array([[age, gender, sepsis_episodes]])
    
    # Apply the same preprocessing as in training
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction_prob = model.predict(input_data_scaled)
    prediction = (prediction_prob > 0.5).astype(int).flatten()[0]
    
    return prediction_prob[0][0], prediction

# Streamlit UI
st.title('Sepsis Survival Prediction')

# Input fields
age = st.number_input('Age', min_value=0, max_value=120, value=30)
gender = st.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Female' if x == 1 else 'Male')
sepsis_episodes = st.number_input('Number of Sepsis Episodes', min_value=0, max_value=10, value=0)

# Predict button
if st.button('Predict'):
    prob, result = predict_survival(age, gender, sepsis_episodes)
    st.write(f'Probability of Survival: {prob:.2f}')
    st.write(f'Prediction: {"Alive" if result == 1 else "Dead"}')
