# -*- coding: utf-8 -*-
"""
Created on Wed May 27 21:11:52 2020

@author: Nicholas
"""


import statsmodels.api as smf
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shap
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from sklearn.linear_model import LogisticRegression, LinearRegression
from tensorflow.keras.models import Sequential
from scipy.stats import linregress
from sklearn.preprocessing import LabelEncoder as le
from sklearn.preprocessing import MinMaxScaler as scaler ## Scale numericals
from sklearn.preprocessing import OneHotEncoder as ohe ### For categorical var
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
pca = PCA()
ohe = ohe()
scaler = scaler(feature_range = (0,1))
le = le()
### Import Datasets ###
df_x = pd.read_csv('Earthquake_damage_training..csv')
df_x.describe()
df_x.info()
df_y=pd.read_csv('Earthquake_damage_labels.csv')
df_x.isnull().sum()#Check for null values
df_y.isnull().sum()
df_y.damage_grade.value_counts()#Check target variables type, multinomial class.
# No null values in the dataset, very good
df_x.has_secondary_use_gov_office.value_counts()

plot_size = plt.rcParams['figure.figsize']
plot_size[0] = 10
plot_size[1] = 14
df_y.damage_grade.value_counts().plot(kind = 'pie', autopct = '%0.05f%%', 
                                      colors = ['lightblue', 'blue', 'red'],
                                      explode = (0.5,0.5,0.5))
'''
Training dataset is made up of numerous binary variables, target data ranges
from 1 (least damage) to 3 (near total destruction)
'''
df_x.geo_level_2_id.value_counts()

categorical_features = ['land_surface_condition','foundation_type','roof_type',
                        'ground_floor_type','other_floor_type','position',
                        'plan_configuration','legal_ownership_status']
df_x.dtypes

### Get dummies ###
land_condition = pd.get_dummies(df_x.land_surface_condition, prefix = 'surface')
found_type = pd.get_dummies(df_x.foundation_type, prefix = 'found')
roof_type = pd.get_dummies(df_x.roof_type, prefix = 'roof')
ground_floor = pd.get_dummies(df_x.ground_floor_type, prefix = 'ground_floor')
other_floor = pd.get_dummies(df_x.other_floor_type, prefix = 'other_floor')
position = pd.get_dummies(df_x.position, prefix = 'pos')
plan = pd.get_dummies(df_x.plan_configuration, prefix = 'plan')
ownership = pd.get_dummies(df_x.legal_ownership_status, prefix = 'ownership')
geo_id = pd.get_dummies(df_x.geo_level_1_id, prefix = 'geo')
x_cat = pd.concat([geo_id,land_condition, found_type, roof_type, ground_floor,
                   other_floor, position, plan, ownership], axis = 1)

### Faster way to get dummies for categorical variables ###
categorical_features = ['land_surface_condition','foundation_type','roof_type',
                        'ground_floor_type','other_floor_type','position',
                        'plan_configuration','legal_ownership_status']
df_categorical = df_x[categorical_features]
df_categorical_x = pd.get_dummies(df_categorical)
df_categorical = pd.concat([df_categorical_x, geo_id], axis = 1) # regions cat
### Scale Numeric columns, codebook says several have already been normalized###
numeric_cols = ['count_floors_pre_eq', 'age', 'area_percentage',
                'height_percentage', 'count_families']
df_x_numeric = df_x[numeric_cols]

df_x_numeric_scaled = scaler.fit_transform(df_x_numeric)
df_x_numeric_scaled = pd.DataFrame({'floors_count':df_x_numeric_scaled[:,0],
                                    'age':df_x_numeric_scaled[:,1],
                                    'area_percent':df_x_numeric_scaled[:,2],
                                    'height_percent':df_x_numeric_scaled[:,3],
                                    'families':df_x_numeric_scaled[:,4]})

# remove old columns from the features
df_x.drop(categorical_features, axis = 1,inplace = True)
df_x.drop(numeric_cols, axis = 1, inplace = True)
X = pd.merge(df_x,df_x_numeric_scaled, left_index = True, right_index = True)
X=  pd.merge(X, df_categorical,left_index = True, right_index = True)
columns_drop = [ 'geo_level_2_id',
                'geo_level_3_id']
X.drop(columns_drop, axis = 1, inplace = True)
X.dtypes
X.set_index('building_id', inplace = True)
X = X.values
y = df_y.damage_grade.values
y = le.fit_transform(y)
from tensorflow.keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1,
                                                    random_state = 1)

y_for_f1_test = y_test
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
from tensorflow.keras.layers import Input
model = Sequential()
model.add(Dense(100, input_shape = (X.shape[1],), activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(50,activation ='relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(15, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(y.shape[1], activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', 
              metrics = ['acc'])
print(model.summary())
history = model.fit(X_train, y_train, batch_size = 1024, epochs = 15, 
                    verbose = 1, validation_split = 0.2)
scores = model.evaluate(X_test, y_test)
print(scores)


"""
According to this model, our deep learning algorithm performs 10 percentage
points better than a baseline of claiming that everything received some damage.
Might be helpful to go through the regional variables using PCA and keep only
some values
"""
### Predicting with our test set

### Import testing data###
df_test_x = pd.read_csv('Earthquake_damage_test.csv')
df_test_x_cat = df_test_x[categorical_features]
geo_id_test = pd.get_dummies(df_test_x.geo_level_1_id)
df_test_x_cat = pd.get_dummies(df_test_x_cat)

df_test_cat = pd.concat([df_test_x_cat, geo_id_test], axis = 1)
### Scale the numeric columns in testing dataframe
df_test_numeric = df_test_x[numeric_cols]

df_test_numeric_scaled = scaler.fit_transform(df_test_numeric)

df_test_numeric_scaled = pd.DataFrame({'floors_count':df_test_numeric_scaled[:,0],
                                    'age':df_test_numeric_scaled[:,1],
                                    'area_percent':df_test_numeric_scaled[:,2],
                                    'height_percent':df_test_numeric_scaled[:,3],
                                    'families':df_test_numeric_scaled[:,4]})
df_test_x.drop(categorical_features, axis = 1, inplace = True)
df_test_x.drop(numeric_cols, axis = 1, inplace = True)
X_pred = pd.merge(df_test_x, df_test_numeric_scaled, left_index = True,
                  right_index = True)
X_pred = pd.merge(X_pred, df_test_cat, left_index = True, right_index = True)
X_pred.drop(columns_drop, axis = 1, inplace = True)
X_pred.set_index('building_id', inplace = True)
X_pred = X_pred.values

predictions = model.predict(X_pred)
predictions = np.argmax(predictions, axis =1)
predictions = pd.Series(predictions)
building_id = df_test_x.building_id
submission = pd.concat([building_id, predictions], axis = 1)
submission.rename(columns = {0 : 'damage_grade'}, inplace = True)
submission.set_index('building_id', inplace = True)
submission.to_csv('Submission.csv')
