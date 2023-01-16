import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load your trained model
model = RandomForestRegressor()
model.load('models/trained_model.pkl')

st.title('AutoValuator: Car Price Prediction')

# Create a function to take user inputs and make a prediction
def predict_price(make, model, year, mileage):
    input_data = {'make': make, 'model': model, 'year': year, 'mileage': mileage}
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    return prediction

# Get user inputs
make = st.text_input('Enter the make of the car')
model = st.text_input('Enter the model of the car')
year = st.text_input('Enter the year of the car')
mileage = st.text_input('Enter the mileage of the car')

# Show the prediction
if st.button('Predict'):
    prediction = predict_price(make, model, year, mileage)
    st.success(f'The predicted price of this car is ${prediction:.2f}')
