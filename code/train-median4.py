import pandas as pd
import numpy as np
import collections
import math
from tempfile import mkdtemp
from shutil import rmtree
import sklearn
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import joblib
from sklearn.model_selection import LeaveOneOut
import sys
import argparse
from datetime import date
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

today = date.today()

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--metric', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--cv', type=int, required=True)
args = parser.parse_args()

name = str(today) + "_" + args.dataset + "_" + str(args.seed) + "_" + args.metric + "_"
if args.cv == 5:
    name = name + "5-fold"
elif args.cv == -1:
    name = name + "loocv"

name = name + "_median"

master = pd.read_csv('data/' + args.dataset + '.csv', index_col=[0])

li = pd.Series([])
ret = []
i = 0

X_train = []
X_test = []
y_train = []
y_test = []

def pretrain(dataset):
    
    global X_train
    global X_test
    global y_train
    global y_test
    global name
    global args
        
    print("Dataset " + name)
    
    ## split
    
    y = dataset['iucn']
    dataset = dataset.drop(['iucn'], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.25, shuffle = True, random_state=args.seed)
    
    tmp = []
    for j in y_train:
        if j == 'LC':
            tmp.append(0)
        else:
            tmp.append(1)
    y_train = tmp
    
    tmp = []
    for j in y_test:
        if j == 'LC':
            tmp.append(0)
        else:
            tmp.append(1)
    y_test = tmp
    

    X_test = X_test[X_train.columns]




def train(dataset):
    
    global X_train
    global X_test
    global y_train
    global y_test
    global name

    ## fit model
    
    cachedir = mkdtemp()

    pipe = Pipeline([
        ("imp", SimpleImputer()),
        ('rf', RandomForestClassifier())
    ], memory = cachedir)
    param_grid = {
        'rf__n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
        'rf__max_features': ['sqrt', 'log2'],
        'rf__max_depth': [None, 3]
    }

    iters = 40

    scoring = ''
    if args.metric == 'roc':
        scoring = 'roc_auc'
    if args.metric == 'prc':
        scoring = 'average_precision'
    if args.metric == 'acc':
        scoring = 'accuracy'

    if args.cv == 5:
        search = RandomizedSearchCV(pipe, param_grid, n_iter = iters, scoring = scoring, cv = 5, n_jobs = -1)
    elif args.cv == -1:
        search = RandomizedSearchCV(pipe, param_grid, n_iter = iters, scoring = scoring, cv = LeaveOneOut(), n_jobs = -1)
    
    search.fit(X_train, y_train)

    print("Best score: " + str(search.best_score_))
    print("Best parameters: " + str(search.best_params_))
    print()
    
    rmtree(cachedir)
    
    ## evaluate model
    
    if args.metric == 'roc':
        metrics.plot_roc_curve(search, X_test, y_test)
        plt.savefig(name + '.png')
        plt.show()

        f = open(name + '.txt', "w")
        f.write('roc: ' + str(search.score(X_test, y_test)))
        f.close()
    if args.metric == 'prc':
        metrics.plot_precision_recall_curve(search, X_test, y_test)
        plt.savefig(name + '.png')
        plt.show()

        f = open(name + '.txt', "w")
        f.write('prc: ' + str(search.score(X_test, y_test)))
        f.close()
    if args.metric == 'acc':
        f = open(name + '.txt', "w")
        f.write('accuracy: ' + str(search.score(X_test, y_test)))
        f.close()

        metrics.plot_precision_recall_curve(search, X_test, y_test)
        plt.savefig(name + '_prc-curve.png')
        plt.show()

        metrics.plot_roc_curve(search, X_test, y_test)
        plt.savefig(name + '_roc-curve.png')
        plt.show()


    ## save model
    
    joblib.dump(search, "MODELS/" + name + ".pkl")

pretrain(master)
train(master)