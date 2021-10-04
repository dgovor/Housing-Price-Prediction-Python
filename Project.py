# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression

pd.options.mode.chained_assignment = None

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train['is_train']  = True
test['is_train'] = False

data_set = pd.concat([train, test], axis=0,sort=False)

missing = data_set.isna().sum()*100/data_set.shape[0]
missing = missing[missing > 50]

for i in range(0,missing.shape[0]):
    data_set = data_set.drop([missing.index[i]],axis=1)
    
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

data_set = data_set.drop(['Street','Utilities','Condition2','RoofMatl','Heating','LowQualFinSF','3SsnPorch','PoolArea'], axis=1)
data_set = data_set.drop([249,313,335,378,581,691,706,934,1061,1182,1190,1298])

data_set.fillna(0,inplace=True) #!!!!!!!!!!!!!!!!!!!!!! change

data_set_tr = data_set.iloc[0:1251,1:68]
data_set_tr_label = data_set.iloc[0:1251,68]

data_set_te = data_set.iloc[1251:1448,1:68]
data_set_te_label = data_set.iloc[1251:1448,68]

data_set_predict = data_set.loc[data_set['is_train'] == False]

features = ['LotArea','BsmtFinSF1','BsmtFinSF2','2ndFlrSF','WoodDeckSF','OpenPorchSF','EnclosedPorch','ScreenPorch','MiscVal']

data_set_tr[features] = (data_set_tr[features]-data_set_tr[features].min())/(data_set_tr[features].max()-data_set_tr[features].min())
data_set_te[features] = (data_set_te[features]-data_set_te[features].min())/(data_set_te[features].max()-data_set_te[features].min())
data_set_predict[features] = (data_set_predict[features]-data_set_predict[features].min())/(data_set_predict[features].max()-data_set_predict[features].min())

data_set_tr['ones'] = 1

w_hat = data_set_tr.T.dot(data_set_tr)+ 0.01*np.eye(68)
w_hat = pd.DataFrame(np.linalg.pinv(w_hat.values), w_hat.columns, w_hat.index)
w_hat = w_hat.T.dot(data_set_tr.T.dot(data_set_tr_label))

w = w_hat[0:67]
b = w_hat[67]

Y = w.T.dot(data_set_te.T) + b

RMSLE = np.sqrt(mean_squared_log_error(data_set_te_label,Y))
error = np.linalg.norm(data_set_te_label-Y)/np.linalg.norm(data_set_te_label)

# model = LinearRegression()
# model.fit(data_set_tr,data_set_tr_label)

# predictions = model.predict(data_set_te)

# error = np.linalg.norm(data_set_te_label-predictions)/np.linalg.norm(data_set_te_label)
