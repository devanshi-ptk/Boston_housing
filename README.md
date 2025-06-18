# Boston Housing Price Prediction using Linear Regression

This project demonstrates the application of a Linear Regression model using the Boston Housing dataset. The model predicts the **median value of owner-occupied homes** based on various housing and environmental features.

## Dataset

The dataset used is `BostonHousing.csv`, which contains housing data for areas in Boston. It includes 11 features and 1 target variable:

### The 11 regressors/ features are:
- `CRIM`: Per capita crime rate by town
- `ZN`: Proportion of residential land zoned for lots over 25,000 sq. ft
- `INDUS`: Proportion of non-retail business acres per town
- `CHAS`: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- `NOX`: Nitric oxide concentration (parts per 10 million)
- `RM`: Average number of rooms per dwelling
- `AGE`: Proportion of owner-occupied units built prior to 1940
- `DIS`: Weighted distances to five Boston employment centers
- `RAD`: Index of accessibility to radial highways
- `TAX`: Full-value property tax rate per $10,000
- `PTRATIO`: Pupil-teacher ratio by town

### The response/ target variable is:
- `MEDV`: Median value of owner-occupied homes in $1000s

## Requirements
- Python 3.x
- pandas
- scikit-learn
- matplotlib

