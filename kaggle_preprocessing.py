
# 1. LOADING LIBRARIES AND DATA 
# 1.1 Libraries 
# Essentials
import numpy as np
import pandas as pd

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Stats
import scipy.stats as stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from math import sqrt
from sklearn.metrics import mean_squared_error

# Models
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import Pool
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor

# Misc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import make_pipeline

# Warnings
import warnings
warnings.filterwarnings('ignore')

# 1.2 Dataset 
# Loading data
house_train=pd.read_csv('train.csv')
house_test=pd.read_csv('test.csv')
data_info=[(house_train.shape[0], house_train.shape[1]), 
           (house_test.shape[0], house_test.shape[1])]

data_shape=pd.DataFrame(data_info, columns=['Rows', 'Columns'], index=['Train', 'Test'])
print(data_shape)


# 2. MANAGING DUPLICATES AND NULL VALUES 
# 2.1 Duplicates 

def mostrar_duplicados(df):
    # Calcula la cantidad de duplicados
    duplicates = df.duplicated().sum()
dups_info=[{'Number of duplicates':mostrar_duplicados(house_train), 
            'Number of duplicates':mostrar_duplicados(house_test)}]

dups_info = pd.DataFrame(dups_info, index=['Train', 'Test'])
print(dups_info)

# 2.2 Null values 

def nulls_values(df):
    df_nulls=pd.DataFrame(df.isna().sum(), columns=['Nulls'])
    df_nulls['Percentage (%)'] = (df_nulls['Nulls']/len(df))*100
    df_nulls=df_nulls[df_nulls['Nulls']>0].sort_values(by='Nulls', ascending=False)
    return df_nulls

nulls_values(house_train).head(10)
nulls_values(house_test).head(10)


train = house_train.drop(columns=["Id"])
test = house_test.drop(columns=["Id"])

# Dealing with null values - training dataset 
fill_with_NA=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']
fill_with_mean=['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
fill_with_mode=['MasVnrType', 'Electrical']

for i in fill_with_NA:
  train[i]=train[i].fillna("NA")

for i in fill_with_mean:
  train[i]=train[i].fillna(train[i].mean())

for i in fill_with_mode: 
  train[i]=train[i].fillna(train[i].value_counts().idxmax())

# Dealing with null values - test dataset 
fill_with_NA=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']
fill_with_mean=['LotFrontage', 'MasVnrArea', 'GarageArea', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1', 'GarageYrBlt', 'BsmtHalfBath', 'BsmtFullBath', 'GarageCars']
fill_with_mode=['MasVnrType', 'MSZoning', 'Functional', 'Utilities', 'KitchenQual', 'Exterior2nd', 'Exterior1st', 'SaleType']

for i in fill_with_NA:
  test[i]=test[i].fillna("NA")

for i in fill_with_mean:
  test[i]=test[i].fillna(test[i].mean())

for i in fill_with_mode: 
  test[i]=test[i].fillna(test[i].value_counts().idxmax())

# Checking null values 
train.isna().sum().sum()
test.isna().sum().sum()

# 3. ORDINAL ENCODING 
ordinal_label = ['LotShape', 'Utilities', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish',
                'GarageQual', 'GarageCond', 'PoolQC', 'Fence']
ordinal_cats = [['Reg', 'IR1', 'IR2', 'IR3'], ['AllPub', 'NoSewr', 'NoSeWa', 'ELO'], ['Gtl', 'Mod', 'Sev'], ['Ex', 'Gd', 'TA', 'Fa', 'Po'], ['Ex', 'Gd', 'TA', 'Fa', 'Po'], ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
                ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'], ['Gd', 'Av', 'Mn', 'No', 'NA'], ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'], ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'], ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
                ['Ex', 'Gd', 'TA', 'Fa', 'Po'], ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'], ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'], ['Fin', 'RFn', 'Unf', 'NA'], ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
                ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'], ['Ex', 'Gd', 'TA', 'Fa', 'NA'], ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NA']]

oe = OrdinalEncoder(categories=ordinal_cats)
train[ordinal_label] = oe.fit_transform(train[ordinal_label])
test[ordinal_label] = oe.fit_transform(test[ordinal_label])


# # 4. NEW FEATURES 
# 4.1 Training dataset 
train['Porch_Area'] = (train[['OpenPorchSF','EnclosedPorch', '3SsnPorch', 'ScreenPorch']]).sum(axis=1)
train['Floor_Sq_Feet'] = (train[['1stFlrSF','2ndFlrSF']]).sum(axis=1)
train['OvrQuality'] = (train[['OverallQual','OverallCond']]).mean(axis=1)
train['ExtQuality'] = (train[['ExterQual','ExterCond']]).mean(axis=1)
train['Bath'] = (train[['FullBath','HalfBath']]).sum(axis=1)
train['BsmtBath'] = (train[['BsmtFullBath', 'BsmtHalfBath']]).sum(axis=1)
train['BsmtFinSF'] = train['TotalBsmtSF'] - train['BsmtUnfSF']
train['BsmtRating'] = (train[['BsmtFinType1','BsmtFinType2']]).mean(axis=1)
train['GarageQuality'] = (train[['GarageQual','GarageCond']]).mean(axis=1)
# 4.2 Test dataset 
test['Porch_Area'] = (test[['OpenPorchSF','EnclosedPorch','3SsnPorch', 'ScreenPorch']]).sum(axis=1)
test['Floor_Sq_Feet'] = (test[['1stFlrSF','2ndFlrSF']]).sum(axis=1)
test['OvrQuality'] = (test[['OverallQual','OverallCond']]).mean(axis=1)
test['ExtQuality'] = (test[['ExterQual','ExterCond']]).mean(axis=1)
test['Bath'] = (test[['FullBath','HalfBath']]).sum(axis=1)
test['BsmtBath'] = (test[['BsmtFullBath', 'BsmtHalfBath']]).sum(axis=1)
test['BsmtFinSF'] = test['TotalBsmtSF'] - test['BsmtUnfSF']
test['BsmtRating'] = (test[['BsmtFinType1','BsmtFinType2']]).mean(axis=1)
test['GarageQuality'] = (test[['GarageQual','GarageCond']]).mean(axis=1)

# 4.3 Dropping features 
vars_to_drop = ['OpenPorchSF','EnclosedPorch','3SsnPorch', '3SsnPorch', 'ScreenPorch', '1stFlrSF', '2ndFlrSF',
                'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF',
                'BsmtUnfSF', 'FullBath', 'HalfBath', 'GarageQual', 'GarageCond', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1',
                'BsmtFinType2']

train=train.drop(vars_to_drop, axis=1)
test=test.drop(vars_to_drop, axis=1)

# # 5. TARGET VISUALIZATION 
# 5.1 Pre-transformation 

import matplotlib.pyplot as plt

sns.set_theme()
plt.figure(figsize=(12,3))
sns.histplot(data=train , x=train['SalePrice'] , kde=True, color='green')
plt.title(f'Distribution of SalesPrice (PreTransformation)', fontsize=15, fontweight='bold')
plt.show()

# 5.2 Post-transformation 
train['SalePrice']=np.log1p(train['SalePrice'])

sns.set_theme()
plt.figure(figsize=(12,3))
sns.histplot(data=train , x=train['SalePrice'] , kde=True, color='green')
plt.title(f'Distribución de SalePrice (post-transformación)', fontsize=15, fontweight='bold')
plt.show()

# 6. SKEWED FEATURES 
# 6.1 Training dataset 
numeric_train=train.dtypes[train.dtypes != "object"].index
numeric_train=numeric_train.to_list()
numeric_train.remove("SalePrice")
numeric_train=pd.Index(numeric_train)
skewed_feats_train=train[numeric_train].apply(lambda x: skew(x)).sort_values(ascending=False)
skewness_train=pd.DataFrame({'Sesgo':skewed_feats_train})
skewness_train.head()
# Sesgo
# Utilities	38.170678
# MiscVal	24.451640
# PoolArea	14.813135
# LotArea	12.195142
# LowQualFinSF	9.002080

high_skewness_train = skewness_train[skewness_train > 0.5]
high_skewness_values = high_skewness_train.index
skew_lambda = 0.15
for feat in high_skewness_values:
    train[feat] = boxcox1p(train[feat], skew_lambda)
    
# 6.2 Test dataset 
numeric_test = test.dtypes[test.dtypes != "object"].index
skewed_feats_test = test[numeric_test].apply(lambda x: skew(x)).sort_values(ascending=False)
skewness_test = pd.DataFrame({'Sesgo':skewed_feats_test})
skewness_test.head()

# Sesgo
# PoolArea	20.176117
# MiscVal	20.054543
# LowQualFinSF	16.150628
# Functional	4.998586
# LandSlope	4.963280

high_skewness_test = skewness_test[skewness_test > 0.5]
high_skewness_values2 = high_skewness_test.index
skew_lambda = 0.15
for feat in high_skewness_values2:
    test[feat] = boxcox1p(test[feat], skew_lambda)

    
# 7. FINAL DATASETS 
# 7.1 Training dataset 
num_train_vars = list(train.select_dtypes(include=['float', 'int']))
cat_train_vars = list(train.select_dtypes(include=['object']))

train_num = train[num_train_vars]
train_cat = train[cat_train_vars]

train = pd.concat([train_num, train_cat], axis=1)
train.shape
# (1460, 67)

# 7.2 Test dataset 
num_test_vars = list(test.select_dtypes(include=['float', 'int']))
cat_test_vars = list(test.select_dtypes(include=['object']))

test_num = test[num_test_vars]
test_cat = test[cat_test_vars]

test = pd.concat([test_num, test_cat], axis=1)
test.shape
# (1459, 66)

# 8. ONE-HOT ENCONDING 
# # Introduction (I)
one_hot = ['MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 
           'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'Electrical', 'GarageType', 'PavedDrive', 'MiscFeature', 'SaleType', 'SaleCondition']
train_test = pd.concat((train,test))
train_test.shape
# (2919, 67)
# # Introduction (II)
ohe = OneHotEncoder()
ohe.fit(train_test[one_hot])

# OneHotEncoder
OneHotEncoder()
# 8.1 Training dataset 
encoded_data = ohe.transform(train[one_hot]).toarray()
onehot_train = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(one_hot))
train = pd.concat([train, onehot_train], axis=1).drop(columns=one_hot)
train.shape
# (1460, 214)
# 8.2 Test dataset 
encoded_data2 = ohe.transform(test[one_hot]).toarray()
onehot_test = pd.DataFrame(encoded_data2, columns=ohe.get_feature_names_out(one_hot))
test = pd.concat([test, onehot_test], axis=1).drop(columns=one_hot)
test.shape
# (1459, 213)
