# Dataset: https://archive.ics.uci.edu/ml/datasets/Abalone
# Task is to predict age of abalone basing on given physical features.
# Since age is numerical value, task was done in 2 ways:
# classification and regression (chose mode in config.yml).

import yaml
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, f1_score
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

config_path = 'D:/scikit/scikit-project-env/config/'
with open(os.path.join(config_path,'config.yml')) as c:
    configs = yaml.safe_load(c)
configs = configs['Abalone age']

col_names = ['Sex','Length','Diameter','High','Whole weigth','Shucked weigth','Viscera weigth','Shell weigth','Rings']
df = pd.read_csv(configs['data_path'],delimiter=',',names=col_names)

# Column transformer

num_features = col_names[1:-1]
cat_features = col_names[:1]

num_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[('num',num_transformer,num_features),('cat',cat_transformer,cat_features)])

# Dataset

target_col = 'Rings'
y = df[target_col]
X = df.drop(columns=[target_col])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, shuffle=True)

# If particular modes are chosen

if configs['mode_clf'] == 1:

    # Choose model

    models = {'knn' : KNeighborsClassifier(n_neighbors=configs['k']),
              'log_reg' : linear_model.LogisticRegression(max_iter=1000)}
    model = models[configs['clf_model']]
    classifier = Pipeline(steps=[('preprocessor',preprocessor),('model',model)])

    # Train

    classifier.fit(X_train,y_train)

    # Evaluate

    score = classifier.score(X_test,y_test)
    preds = classifier.predict(X_test)
    acc = accuracy_score(y_test,preds)
    prec = precision_score(y_test,preds,average='macro',zero_division=0)
    recall = recall_score(y_test,preds,average='macro')
    print('---CLASSIFIER---')
    print(f'Score: {score:.4f}. Accuracy {acc:.4f}. Precision {prec:.4f}. Recall {recall:.4f}.')

    cf_matrix = confusion_matrix(y_test,preds)
    print('Confusion matrix: \n',cf_matrix)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=False, fmt='.2%', cmap='PuBu')
    plt.show()

if configs['mode_reg'] == 1:

    # Choose model

    model2 = linear_model.LinearRegression()
    regressor = Pipeline(steps=[('preprocessor',preprocessor),('model',model2)])

    # Train

    regressor.fit(X_train,y_train)

    # Evaluate

    scores = regressor.score(X_test,y_test)
    preds = regressor.predict(X_test)
    rmse = mean_squared_error(y_test.values,preds,squared=True)
    mae = mean_absolute_error(y_test,preds)
    r2score = r2_score(y_test,preds)
    print('---REGRESSOR---')
    print(f'Score: {scores:.4f}. R^2 score: {r2score:.4f}. RMSE: {rmse:.4f}. MAE: {mae:.4f}.')



