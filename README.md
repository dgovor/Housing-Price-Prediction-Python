# Housing-Price-Prediction-MATLAB
Machine Learning (ML) model for price prediction using a combination of Linear Regression, Gradient Boosting Regression, XGBoost Regressor, and LGBM Regressor.

## Description

This code was written in Python for the [competition presented by Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview).
The proposed ML model was developed in order to represent one of the possible solutions for the housing price prediction problem. 

## Dataset

The dataset provided by Kaggle consists of 2919 samples with 79 features each. This dataset originally split into training and testing datasets with 1460 and 1459 samples, respectively. In order to justify our models performance, the training dataset is split using train_test_split into two subsets of data. One subset contains 80% of the original training data and is used to train our model, second subset that is called validation subset contains the remaining 20% and is used to validate our model. The accuracy of validation using these 20% of the training data will provide us with an understanding of the efficiency of our design.

## Data preprocessing

Data preprocessing consists of the following steps:
* The data is cleaned from features that contained more than 50% of missing data;
* All categorical features are transformed into numerical features;
* The features are sorted so it would be possible to describe them linearly;
* Some features with very low variance are deleted;
* Outliers are deleted;
* All missing values are found and changed to either 0 or most frequent values of the features that contain these missing values, wherever it makes sence;
* The data is normalized, where necessary.

## Linear Regression

The training dataset is used to calculate "_w_" and "_b_". This is done by solving the equation _w&#770; = (X&#770;<sup>T</sup> X&#770; - &epsilon;I<sub>68</sub>)<sup>-1</sup>X&#770;<sup>T</sup>y_, where _X&#770;_ is a modified version of the training dataset, _y_ is a vector that contains the labels (prices), and _&epsilon_ is a small value, in our case 0.01. It is important to notice that it is necessary to include a term _&epsilon;I<sub>68</sub>_ in the equation to ensure that the inverse _(X&#770;<sup>T</sup> X&#770; - &epsilon;I<sub>68</sub>)<sup>-1</sup>_ does exist. Otherwise, the matrix can be badly scaled and results may be inaccurate. After solving the equation we can receive optimal parameters "_w<sup>*</sup>_" and "_b<sup>*</sup>_". They are used to create a linear model that is able to predict the prices of the houses based on their features. The prices can be found by solving _Y = w<sup>T</sup>X + b_.

## Results

As a prediction method it was decided to use a linear regression method since the given data can be described linearly. This method turned out to be fairly accurate as it showed a high percentage of accuracy. As one of the evaluation methods, RMSLE is used to calculate error. As another form of evaluation, relative prediction error percentage is used. As the final result RMSLE showed an error of 14%. Relative prediction error percentage was 12%. As it can be seen, the final accuracy of the algorithm is approximately 87%.
