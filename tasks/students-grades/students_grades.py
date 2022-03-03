# Dataset: https://archive.ics.uci.edu/ml/datasets/Student+Performance
# Basing on given features, task is to manually find which of them influence the final grade

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import yaml

config_path = 'D:/scikit/scikit-project-env/config/'
with open(os.path.join(config_path,'config.yml')) as c:
    configs = yaml.safe_load(c)
configs = configs['Students grades']

output_path = configs['output_path']
df = pd.read_csv(configs['data_path'], delimiter=';')     

# All of the features

features = {
    1 : 'G3',       # Final grade
    2 : 'G1',       # First term grade
    3 : 'G2',       # Second term grade
    4 : 'age',
    5 : 'Medu',
    6 : 'Fedu',
    7 : 'traveltime',
    8 : 'studytime',
    9 : 'failures',
    10 : 'famrel',
    11 : 'freetime',
    12 : 'goout',
    13 : 'Dalc',
    14 : 'Walc',
    15 : 'health',
    16 : 'absences',
    17 : 'school',
    18 : 'sex',
    19 : 'address',
    20 : 'famsize',
    21 : 'Pstatus',
    22 : 'Mjob',
    23 : 'Fjob',
    24 : 'reason',
    25 : 'guardian',
    26 : 'schoolsup',
    27 : 'famsup',
    28 : 'paid',
    29 : 'activities',
    30 : 'nursery',
    31 : 'higher',
    32 : 'internet',
    33 : 'romantic'
}

# Choose what columns (features) to include in training

chosen_ftrs = configs['chosen_features']

# Column Transformer to handle mixed numerical and categorical data

num_features =[features[f] for f in chosen_ftrs if f < 17]
cat_features = [features[f] for f in chosen_ftrs if f >= 17]

num_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[('num',num_transformer,num_features),('cat',cat_transformer,cat_features)])

model = linear_model.LinearRegression()
regressor = Pipeline(steps=[('preprocessor',preprocessor),('model',model)])

y = df['G3']
X = df[num_features+cat_features]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,shuffle=True)

# Train model

regressor.fit(X_train,y_train)

# Evaluate

scores = regressor.score(X_test,y_test)
print(f'Coefficient of determination of the prediction: {scores:.4f}')

test = X_test[:1]
pred = regressor.predict(test)
print(f'Sample prediction: {round(pred[0])}, y_test: {y_test[:1].values[0]}.')  #float, int

margin = configs['margin']
if int(round(pred[0])) <= (y_test[:1].values[0]+margin) and int(round(pred[0])) >= (y_test[:1].values[0]-margin) :
    print(f'Prediction within margin +/- {margin}.')
else:
    print(f'Prediction outside of margin +/- {margin}.')

# Metrics

preds = regressor.predict(X_test)
rmse = mean_squared_error(y_test.values,preds,squared=True)
mae = mean_absolute_error(y_test,preds)
r2score = r2_score(y_test,preds)
print(f'RMSE: {rmse:.4f}. MAE: {mae:.4f}. R^2 score: {r2score:.4f}.')

# Write results to file

file = open(os.path.join(output_path,'students_grades_results.txt'),mode='a')
file.write(f'Score {scores:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R^2 score: {r2score:.4f}, X {str(chosen_ftrs)}\n')
file.close()




