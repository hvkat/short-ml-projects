# Task is to train various classifiers on MNIST dataset
# (sklearn's load_digits() [1797 instances] was used instead of full MNIST dataset [10000 instances])

import yaml
import os
from sklearn.datasets import load_digits
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

config_path = 'D:/scikit/scikit-project-env/config/'
with open(os.path.join(config_path,'config.yml')) as c:
    configs = yaml.safe_load(c)
configs = configs['Mnist classification']

# Load data

digits = load_digits()

# Visualize sample data instances

if configs['visualize_samples']:
    n = configs['n_visualize']
    fig,axs = plt.subplots(1,n)
    fig.suptitle('Example data instances')
    for i in range(n):
        j = random.randrange(len(digits.images)+1)
        axs[i].matshow(digits.images[j]), axs[i].set_title(digits.target[j]), axs[i].set_axis_off()
    plt.show()

# Prepare data

print(f'Number of instances in dataset and their shape: {np.shape(digits.images)}')
X = digits.images.reshape(len(digits.images),-1)
print(f'Instances reshaped: {np.shape(X)}')
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=.2)

# Prepare suplementary function for training

def train(model):
    model.fit(X_train,y_train)
    score = model.score(X_test,y_test)
    preds = model.predict(X_test)
    cf_matrix = confusion_matrix(y_test,preds)
    return score, cf_matrix

# Train models

models = {
    'k-nearest neighbors' : KNeighborsClassifier(n_neighbors=configs['KNN']['n_neighbors'],n_jobs=configs['KNN']['n_jobs']),
    'logistic regression' : LogisticRegression(max_iter=configs['LOGREG']['max_iter'],solver=configs['LOGREG']['solver'],verbose=configs['verbose']),
    'multi-layer perceptron': MLPClassifier(hidden_layer_sizes=(configs['MLP']['hls_1'],configs['MLP']['hls_2']),
                                            early_stopping=configs['MLP']['early_stopping'], verbose=configs['verbose'],
                                            n_iter_no_change=configs['MLP']['n_iter_no_change']),
    'support vector machine' : SVC(verbose=configs['verbose'])
}

scores = []
cf_matrices = []
for m in models:
    print(f'Training model: {m}')
    score, cf_matrix = train(models[m])
    scores.append(score), cf_matrices.append(cf_matrix)

# Visualize results

fig, axs = plt.subplots(2,int(len(models)/2))
plt.suptitle('Confusion matrices')
for i in range(len(models)):
    if i<2:
        ax = axs[0][i]
    else:
        ax = axs[1][i-2]
    sns.heatmap(cf_matrices[i]/np.sum(cf_matrices[i]),annot=False,cmap='Blues',ax=ax)           # cmap='mako' for negative colors
    ax.set_title(f'{list(models.keys())[i]}\n score: {scores[i]:.4f}')
    ax.set_axis_off()
plt.show()





