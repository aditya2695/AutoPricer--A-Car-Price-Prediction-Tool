
import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse


import statsmodels.api as sm
import statsmodels.tools 
import os
import pickle
import pandas as pd6
import numpy as np



def eval_metrics(actual, pred, num_features):
    rmse = round(np.sqrt(mean_squared_error(actual, pred)), 3)
    mae = round(mean_absolute_error(actual, pred), 2)
    r2 = round(r2_score(actual, pred), 2)
    adj_r2 = round(1 - (1-r2)*(len(actual)-1)/(len(actual)-num_features-1), 2)
    return rmse, mae, r2, adj_r2

def remove_outliers(df,feature):

    # # Calculate the IQR
    # Q1 = df[feature].quantile(0.25)
    # Q3 = df[feature].quantile(0.75)
    # IQR = Q3 - Q1

    # # Remove outliers
    # df = df[(df[feature] >= Q1 - 1.5*IQR) & (df[feature] <= Q3 + 1.5*IQR)]

    # Compute the z-score for each value in the 'price' column
    z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())

    # Define a threshold for outliers (e.g., z-score > 3)
    threshold = 3

    # Identify the indices of outliers in the 'price' column
    outlier_indices = np.where(z_scores > threshold)

    # Remove the outliers from the 'price' column
    df = df.drop(outlier_indices[0])

    return df



def evaluate_models(X_test,y_test,fitered_cols):
    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['Model', 'RMSE', 'MAE', 'R2'])

    # Define the path to the models folder
    models_folder = 'models/'

    # Iterate over each file in the models folder
    for file_name in os.listdir(models_folder):
        # Check if the file is a model file (ends with '.pkl')
        if file_name.endswith('.pkl'):
            # Load the model from the file

            if file_name.__contains__=='':
                fitered_cols=list(X_test.columns).remove('count')
                X_test=X_test[fitered_cols]
            with open(models_folder + file_name, 'rb') as f:
                model = pickle.load(f)

            # Make predictions on X_test
            y_pred = model.predict(X_test)

            # Calculate evaluation metrics
            rmse, mae, r2 = np.sqrt(eval_metrics(y_test, y_pred))

            # Add the results to the DataFrame
            model_name = file_name[:-4]  # remove '.pkl' from the file name
            model_results = pd.DataFrame({'Model': model_name,
                                           'RMSE': rmse,
                                           'MAE': mae,
                                           'R2': r2},
                                          index=[0])

            # Add the results to the overall DataFrame using pd.concat()
            results_df = pd.concat([results_df, model_results], ignore_index=True)

    return results_df


def predict_price(car_input, model,filtered_cols):
    # Create a DataFrame from the input dictionary
    df = pd.DataFrame([car_input])

    # Extract the car's condition
    condition = df['Condition'][0]

    # Drop columns that are not needed
    cols_to_drop = ['make', 'writeoff', 'Condition']
    df.drop(columns=cols_to_drop, inplace=True, axis=1)

    # One-hot encode categorical columns
    cols_to_ohe = ['model', 'fuel', 'ULEZ', 'transmission', 'body', 'year']
    df_ohe = pd.get_dummies(df, columns=cols_to_ohe)
    df_ohe = sm.add_constant(df_ohe)

    # Reindex the DataFrame to match the filtered_cols used to train the model
    test_ohe = df_ohe.reindex(columns=filtered_cols, fill_value=0)

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
