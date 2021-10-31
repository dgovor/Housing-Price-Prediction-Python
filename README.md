# Housing-Price-Prediction-MATLAB
Machine Learning (ML) model for price prediction using a combination of Linear Regression, Gradient Boosting Regression, XGBoost Regressor, and LGBM Regressor.

## Description

This code was written in Python for the [competition presented by Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview).
The proposed ML model was developed in order to represent one of the possible solutions for the housing price prediction problem. 

## Dataset

The dataset provided by Kaggle consists of 2919 samples with 79 features each. This dataset originally split into training and testing datasets with 1460 and 1459 samples, respectively. In order to justify our models performance, the training dataset is split using train_test_split into two subsets of data. One subset contains 80% of the original training data and is used to train our model, second subset that is called validation subset contains the remaining 20% and is used to validate our model. The accuracy of validation using these 20% of the training data will provide us with an understanding of the efficiency of our design.

## Data Preprocessing

Data preprocessing consists of the following steps:
* The data is cleaned from features that contained more than 50% of missing data;
* All categorical features are transformed into numerical features;
* The features are sorted so it would be possible to describe them linearly;
* Some features with very low variance are deleted;
* Outliers are deleted;
* All missing values are found and handled;
* The data is normalized.

## Methods for Regression

The training subset is used to train our model that consists of the following methods:
* Linear Regression;
* Gradient Boosting Regression;
* XGBoost Regressor;
* LGBM Regressor.

## Results

To obtain the results it was decided to combine all of the methods mentioned before into a single ensemble. In this ensemble the methods have different weights: Linear Regression (20%), Gradient Boosting Regression (15%), XGBoost Regressor (30%), and LGBM Regressor (35%). This ensemble provided accurate results as it showed a low mean absolute percentage error of approximately 7%. Which suggests that the final accuracy of the model is approximately 93%.
