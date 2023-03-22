# AutoPricer: A Car Price Prediction Tool



### Introduction



This project aims to predict the prices of used cars based on their features. The predictions are made using regression techniques on a dataset scraped from Autotrader, a popular website for buying and selling cars in the UK. The project includes a Streamlit app that allows users to input the features of a car and get a predicted price.

<div style="display: flex;">
  <img  src="https://images.pexels.com/photos/243206/pexels-photo-243206.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2" alt="Image 1" style="width: 50%;">
  <img src="https://m.atcdn.co.uk/ect/media/%7Bresize%7D/6261c7bb2152499fb8c6006c9b8497c4.jpg" alt="Image 2" style="width: 50%;">
</div>


### Dataset

The dataset used in this project was scraped from Autotrader using Python and the Beautiful Soup library. The data includes information about various cars, such as their make, model, year, mileage, fuel type, transmission, and more. The dataset was cleaned and preprocessed using Pandas and NumPy to remove any missing values and ensure that the data is in the correct format for regression analysis.

### Regression Techniques

This project uses two regression techniques to predict car prices: Linear Regression and XGBoost. Linear Regression is a simple and interpretable model that fits a linear relationship between the input features and the output variable (i.e., the price). XGBoost is a more complex and powerful model that uses gradient boosting to improve the accuracy of the predictions.

The models were trained on the cleaned dataset using scikit-learn and XGBoost libraries. The models were evaluated using the mean absolute error (MAE) and mean squared error (MSE) metrics to determine their accuracy.

### Evaluation

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>R-squared</th>
      <th>Adjusted R-squared</th>
      <th>Root mean squared error</th>
      <th>Mean absolute error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Linear regression</td>
      <td>0.926</td>
      <td>0.926</td>
      <td>1244</td>
      <td>-</td>
    </tr>
    <tr>
      <td>XgB Regression</td>
      <td>0.95</td>
      <td>6.7</td>
      <td>1051</td>
      <td>750</td>
    </tr>
    
  </tbody>
</table>


### Getting Started

To run the project, you will need to install the following libraries:

    Beautiful Soup
    Pandas
    NumPy
    scikit-learn
    XGBoost
    Streamlit

To scrape the Autotrader website and preprocess the data, run the data_scraper.py and data_cleaning.py scripts, respectively. To train the regression models and evaluate their accuracy, run the linear_regression.py and xgboost_regression.py scripts.