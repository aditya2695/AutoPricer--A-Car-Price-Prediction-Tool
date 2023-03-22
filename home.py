import streamlit as st
import pickle
import pandas as pd
import numpy as np
import datetime

def predict_price(car_input, model,filtered_cols):
    # Create a DataFrame from the input dictionary
    df = pd.DataFrame([car_input])

    # df['model'] = df['model'].str.split(' ').str[1]

    df['model'] = df['model'].str.upper()

    current_year = datetime.datetime.now().year

    df['year'] = df['year'].astype('int')
    df['age'] = np.where(pd.notnull(df['year']), current_year - df['year'], df['year'].max())
    df.drop(['year'], axis=1, inplace=True)
    
    # Extract the car's condition
    condition = df['Condition'][0]

    # Drop columns that are not needed
    cols_to_drop = ['Condition']
    df.drop(columns=cols_to_drop, inplace=True, axis=1)
    df['ULEZ'] = df['ULEZ'].apply(lambda x: 1 if x == 'ULEZ' else 0)

    # One-hot encode categorical columns
    cols_to_ohe = ['model', 'fuel', 'transmission', 'body']
    df_ohe = pd.get_dummies(df, columns=cols_to_ohe)
    

    # Reindex the DataFrame to match the filtered_cols used to train the model
    test_ohe = df_ohe.reindex(columns=filtered_cols, fill_value=0)

    print(test_ohe)

    # Make a prediction using the model
    price = model.predict(test_ohe)[0]

    # Apply a discount based on the car's condition
    if condition.lower() == 'no damage':
        price *= 0.8
    elif condition.lower() == 'moderate damage':
        price *= 0.5
    elif condition.lower() == 'heavy damage':
        price *= 0.3

    # Format the price as a string and return it
    return f"Predicted price for {car_input['model']} is £{round(price, 2)}"

# car_input = {'make': 'Vauxhall',
#              'model':'Grandland',
#              'year':'2015',
#              'writeoff':'',
#              'mileage':4000,
#              'BHP':100,
#              'transmission':'Manual',
#              'fuel':'Petrol',
#              'owners':2,
#              'body':'Hatchback',
#              'ULEZ':'Yes',
#              'engine':1.4,
#              'Condition':'heavy damage',
#             }


# {'model': 'Astra',
#  fuel:'Petrol',
# 'mileage': 100,
# 'BHP': 100,
# 'owners': 2,
# 'engine': 1.5,
# 'ULEZ': 1,
# 'body': 'Hatchback',
# 'damage': 'Heavy damage',
# 'year': 2023}

def main():
    # Load the saved model
    with open('models/xgb_model.pkl', 'rb') as f:
        xgb_reg = pickle.load(f)

    # Define the features used to train the model
    filtered_cols = ['mileage', 'BHP', 'owners', 'engine']

    # Create a form to input the car details
    st.title('Car Price Predictor')
    car_input = {}

    car_input['model']  = st.selectbox('Model', ['Astra', 'Mokka','Grandland'])
    car_input['fuel']  = st.selectbox('Fuel', ['Petrol','Diesel'])
    car_input['mileage'] = st.number_input('Mileage', value=0)
    car_input['BHP'] = st.number_input('BHP', value=100)
    car_input['owners'] = st.number_input('Owners', value=0)
    car_input['engine'] = st.number_input('Engine', value=1.5)
    car_input['ULEZ'] = 1 if st.radio('ULEZ', ['Yes', 'No']) == 'Yes' else 0
    car_input['body'] = st.selectbox('body', ['Hatchback', 'Estate'])
    car_input['damage'] = st.selectbox('Damage', ['Heavy damage', 'moderate damage','no damage'])
    car_input['year']  = st.number_input('Year', value=2023)

    print(car_input)

    # Add more input fields as needed

    # Add a button to predict the price
    if st.button('Predict Price'):
        predicted_price = predict_price(car_input, xgb_reg, filtered_cols)
        st.write('Predicted Price:', predicted_price)

if __name__ == '__main__':
    main()
