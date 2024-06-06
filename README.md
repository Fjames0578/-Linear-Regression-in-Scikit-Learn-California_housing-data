# -Linear-Regression-in-Scikit-Learn-California_housing-data
Linear Regression 
Using this data, our model should be able to predict the value of a house using the features given in the dataset.

# Loading the Dataset:

The California housing dataset is loaded using fetch_california_housing from sklearn.datasets.
Features and target variables are extracted and stored in DataFrames.
Splitting the Data:

The data is split into training and testing sets using train_test_split with 80% for training and 20% for testing.
Training the Model:

A linear regression model is created using LinearRegression() from sklearn.linear_model.
The model is trained on the training data using the fit method.
Making Predictions:

The trained model is used to make predictions on the test data.
Evaluating the Model:

Model performance is evaluated using Mean Squared Error (MSE) and R-squared (R2) metrics.
The results are printed to the console.
Displaying Coefficients:

The coefficients of the linear regression model are displayed, indicating the impact of each feature on the target variable.
This script provides a straightforward demonstration of building and evaluating a linear regression model to predict housing prices using the California housing dataset.
