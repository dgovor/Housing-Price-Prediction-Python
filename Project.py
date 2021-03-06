# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# A new feature 'is_train' is created to simplify our work with the datasets.
train['is_train']  = True
test['is_train'] = False

# Both datasets are combined together in order to perform data preprocessing.
data_set = pd.concat([train, test], axis=0,sort=False)
data_set.reset_index(drop=True, inplace=True)

# Find all features that have 50% or more of missing values.
missing = data_set.isna().sum()*100/data_set.shape[0]
missing = missing[missing > 50]

# Drop these features since they do not carry a valuable insight.
for i in range(0,missing.shape[0]):
    data_set = data_set.drop([missing.index[i]],axis=1)

# Next lines (35-91) are used to encode all categorical features into numeric ones.
# First, numeric values are assigned in numerical order. Later, we rearange these values 
# in the way so it would be easier to describe them linearly.
data_set['MSZoning'].replace({'A':6,'C (all)':1,'FV':5,'I':7,'RH':3,'RL':4,'RP':8,'RM':2},inplace=True)
data_set['Street'].replace({'Grvl':1,'Pave':2},inplace=True)
data_set['LotShape'].replace({'Reg':1,'IR1':2,'IR2':4,'IR3':3},inplace=True)
data_set['LandContour'].replace({'Lvl':2,'Bnk':1,'HLS':4,'Low':3},inplace=True)
data_set['Utilities'].replace({'AllPub':1,'NoSewr':2,'NoSeWa':3,'ELO':3},inplace=True)
data_set['LotConfig'].replace({'Inside':2,'Corner':3,'CulDSac':5,'FR2':1,'FR3':4},inplace=True)
data_set['LandSlope'].replace({'Gtl':2,'Mod':3,'Sev':1},inplace=True)
data_set['Neighborhood'].replace({'Blmngtn':16,'Blueste':8,'BrDale':3,'BrkSide':4,'ClearCr':19,\
                                  'CollgCr':17,'Crawfor':18,'Edwards':5,'Gilbert':15,'IDOTRR':2,\
                                      'MeadowV':1,'Mitchel':12,'NAmes':11,'NoRidge':25,'NPkVill':10,\
                                          'NridgHt':24,'NWAmes':14,'OldTown':6,'SWISU':9,'Sawyer':7,\
                                              'SawyerW':13,'Somerst':20,'StoneBr':23,'Timber':22,'Veenker':21},inplace=True)
data_set['Condition1'].replace({'Artery':1,'Feedr':3,'Norm':4,'RRNn':7,'RRAn':5,'PosN':8,'PosA':9,'RRNe':6,'RRAe':2},inplace=True)
data_set['Condition2'].replace({'Artery':1,'Feedr':3,'Norm':4,'RRNn':7,'RRAn':5,'PosN':8,'PosA':9,'RRNe':6,'RRAe':2},inplace=True)
data_set['BldgType'].replace({'1Fam':5,'2fmCon':1,'Duplex':2,'TwnhsE':4,'Twnhs':3},inplace=True)
data_set['HouseStyle'].replace({'1Story':6,'1.5Fin':3,'1.5Unf':1,'2Story':7,'2.5Fin':8,'2.5Unf':4,'SFoyer':2,'SLvl':5},inplace=True)
data_set['RoofStyle'].replace({'Flat':4,'Gable':2,'Gambrel':1,'Hip':5,'Mansard':3,'Shed':6},inplace=True)
data_set['RoofMatl'].replace({'ClyTile':1,'CompShg':2,'Membran':3,'Metal':4,'Roll':5,'Tar&Grv':6,'WdShake':7,'WdShngl':8},inplace=True)
data_set['Exterior1st'].replace({'AsbShng':4,'AsphShn':2,'BrkComm':1,'BrkFace':11,'CBlock':3,'CemntBd':13,\
                                 'HdBoard':8,'ImStucc':15,'MetalSd':6,'Other':16,'Plywood':10,'PreCast':17,\
                                     'Stone':14,'Stucco':9,'VinylSd':12,'Wd Sdng':5,'WdShing':7},inplace=True)
data_set['Exterior2nd'].replace({'AsbShng':2,'AsphShn':4,'Brk Cmn':3,'BrkFace':12,'CBlock':1,'CmentBd':14,\
                                 'HdBoard':11,'ImStucc':15,'MetalSd':6,'Other':16,'Plywood':10,'PreCast':17,\
                                     'Stone':8,'Stucco':7,'VinylSd':13,'Wd Sdng':5,'Wd Shng':9},inplace=True)
data_set['MasVnrType'].replace({'BrkCmn':1,'BrkFace':2,'CBlock':3,'None':0,'Stone':4},inplace=True)
data_set['ExterQual'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},inplace=True)
data_set['ExterCond'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},inplace=True)
data_set['Foundation'].replace({'BrkTil':2,'CBlock':3,'PConc':6,'Slab':1,'Stone':4,'Wood':5},inplace=True)
data_set['BsmtQual'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},inplace=True)
data_set['BsmtQual'].fillna(0,inplace=True)
data_set['BsmtCond'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},inplace=True)
data_set['BsmtCond'].fillna(0,inplace=True)
data_set['BsmtExposure'].replace({'Gd':4,'Av':3,'Mn':2,'No':1},inplace=True)
data_set['BsmtExposure'].fillna(0,inplace=True)
data_set['BsmtFinType1'].replace({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1},inplace=True)
data_set['BsmtFinType1'].fillna(0,inplace=True)
data_set['BsmtFinType2'].replace({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1},inplace=True)
data_set['BsmtFinType2'].fillna(0,inplace=True)
data_set['Heating'].replace({'Floor':1,'GasA':6,'GasW':5,'Grav':2,'OthW':4,'Wall':3},inplace=True)
data_set['HeatingQC'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},inplace=True)
data_set['CentralAir'].replace({'N':0,'Y':1},inplace=True)
data_set['Electrical'].replace({'SBrkr':5,'FuseA':4,'FuseF':3,'FuseP':2,'Mix':1},inplace=True)
data_set['KitchenQual'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},inplace=True)
data_set['Functional'].replace({'Typ':8,'Min1':7,'Min2':6,'Mod':5,'Maj1':4,'Maj2':3,'Sev':2,'Sal':1},inplace=True)
data_set['FireplaceQu'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},inplace=True)
data_set['FireplaceQu'].fillna(0,inplace=True)
data_set['GarageType'].replace({'BuiltIn':6,'Attchd':5,'Basment':4,'2Types':3,'Detchd':2,'CarPort':1},inplace=True)
data_set['GarageType'].fillna(0,inplace=True)
data_set['GarageFinish'].replace({'Fin':3,'RFn':2,'Unf':1},inplace=True)
data_set['GarageFinish'].fillna(0,inplace=True)
data_set['GarageQual'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},inplace=True)
data_set['GarageQual'].fillna(0,inplace=True)
data_set['GarageCond'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},inplace=True)
data_set['GarageCond'].fillna(0,inplace=True)
data_set['PavedDrive'].replace({'Y':3,'P':2,'N':1},inplace=True)
data_set['SaleType'].replace({'VWD':10,'New':9,'Con':8,'CWD':7,'ConLI':6,'WD':5,'COD':4,'ConLw':3,'ConLD':2,'Oth':1},inplace=True)
data_set['SaleCondition'].replace({'Partial':6,'Normal':5,'Alloca':4,'Family':3,'Abnorml':2,'AdjLand':1},inplace=True)

# Analysis of the dataset shows that certain features can be disregarded due to their low variance.
data_set = data_set.drop(['Street','Utilities','Condition2','RoofMatl','Heating','LowQualFinSF','3SsnPorch','PoolArea'], axis=1)
# Some of the data samples did not follow the same patern, and therefore, were classified as outliers.
data_set = data_set.drop([249,313,335,378,581,691,706,934,1061,1182,1190,1298])

# Find the remaining missing values.
missing_val = data_set.isna().sum()

data_set['LotFrontage'].fillna(data_set['LotFrontage'].min(),inplace=True) # Missing values of 'LotFrontage' are changed to the min value of this feature.
data_set.fillna(0,inplace=True) # The rest of the missing values are changed to zeros.

# The data is split into training and challenge data.
data_set_predict = data_set.loc[data_set['is_train'] == False]
data_set_predict = data_set_predict.drop(['Id','is_train','SalePrice'],axis=1)
data_set = data_set.loc[data_set['is_train'] == True]

# Find correlation between the features.
# correlation = data_set.corr()
# plt.figure(figsize=(10,10))
# sns.heatmap(correlation, cbar=True, square=True, cmap='Blues')

X = data_set.drop(['Id','is_train','SalePrice'],axis=1)
Y = data_set['SalePrice']

# The dataset is split into training (80%) and testing (20%) subsets.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=7)

# The data is being normalized based on the training subset.
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
data_set_predict = scaler.transform(data_set_predict)

# The predictions are calculated on the basis of four different models.
model1 = XGBRegressor(learning_rate=0.1, subsample=0.71, random_state=4, n_estimators=500)
model1.fit(X_train,Y_train)
predictions1 = model1.predict(X_test)

model2 = LinearRegression()
model2.fit(X_train,Y_train)
predictions2 = model2.predict(X_test)

model3 = GradientBoostingRegressor(learning_rate=0.1, random_state=4)
model3.fit(X_train,Y_train)
predictions3 = model3.predict(X_test)

model4 = LGBMRegressor(learning_rate=0.1, random_state=4)
model4.fit(X_train,Y_train)
predictions4 = model4.predict(X_test)

# The final predictions are calculated based on the following ensemble of the models.
predictions = predictions1*0.3 + predictions2*0.2 + predictions3*0.15 + predictions4*0.35

# Mean absolute percentage error.
error = metrics.mean_absolute_percentage_error(Y_test, predictions)

# The predictions for Kaggle challenge are calculated following the same ensemble.
output_pred = model1.predict(data_set_predict)*0.3 + model2.predict(data_set_predict)*0.2 + model3.predict(data_set_predict)*0.15 + model4.predict(data_set_predict)*0.35

output = pd.DataFrame()
output['Id'] = test['Id']
output['SalePrice'] = output_pred

output.to_csv('submission.csv',index=False)

plt.scatter(predictions, Y_test, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], '--r', linewidth=2)
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('House Price Prediction')
plt.show()
