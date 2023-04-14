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
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import joblib
from sklearn.model_selection import LeaveOneOut
import sys
import argparse
from datetime import date

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

name = name + "_phylo"

master = pd.read_csv('data/' + args.dataset + '.csv', index_col=[0])


phylo = pd.read_csv('raw-data/zoonomia_dataset_17nov2021_streamlined.csv', index_col=0)
phylo = phylo.loc[master.index]
phylo.index.name = None
ids = phylo.index.copy()
genus = [i.split('_')[0] for i in ids]
phylo['Genus'] = genus
phylo = phylo[['Genus', 'Family', 'Order']]
phylo = phylo.reset_index()

li = pd.Series([])
ret = []
i = 0

def helper2(x):
    
    global li
    global ret
    global i
    
    m = x.value_counts().to_dict()[li.tolist()[i]]
        
    bound = 0.75 * (x.count())
        
    if m <= bound:
        ret.append(True)
    else:
        ret.append(False)
    
    i += 1

def helper(df):
    
    global ret
    global i
    ret = []
    i = 0
    df.apply(helper2, axis=0)
    return ret


def clean2(dataset):
    
    global li
    
    li = dataset.mode().loc[0]
    tt = helper(dataset)
    dataset = dataset[dataset.columns[tt]]
    
    return dataset

def clean1(dataset):
    
    nans = dataset.isna().sum()
    bound = 0.75 * len(dataset)
    dd = ([i <= bound for i in nans])
    dataset = dataset[dataset.columns[dd]]
    
    return dataset

ibound = 0

from statistics import median
def med(l):
    if len(l) == 0:
        return math.nan
    r = [i for i in l if not math.isnan(i)]
    return median(r)

def impute(col):
    genus = {}
    family = {}
    order = {}
    
    for j in set(phylo['Order']):
        order[j] = []
    for j in set(phylo['Family']):
        family[j] = []
    for j in set(phylo['Genus']):
        genus[j] = []
        
    for j, val in enumerate(col):
        if not math.isnan(val) and j < ibound:

            order[phylo.loc[j]['Order']].append(val)
            family[phylo.loc[j]['Family']].append(val)
            genus[phylo.loc[j]['Genus']].append(val)

    for j in order:
        order[j] = med(order[j])

    for j in family:
        family[j] = med(family[j])

    for j in genus:
        genus[j] = med(genus[j])

    tmp = col[:ibound].tolist()

    for j, val in enumerate(col):
        if not math.isnan(val):
            col[j] = (val)
        elif not math.isnan(genus[phylo.loc[j]['Genus']]):
            col[j] = (genus[phylo.loc[j]['Genus']])
        elif not math.isnan(family[phylo.loc[j]['Family']]):
            col[j] = (family[phylo.loc[j]['Family']])
        elif not math.isnan(order[phylo.loc[j]['Order']]):
            col[j] = (order[phylo.loc[j]['Order']])
        else:
            col[j] = (med(tmp))

def rnan(dataset):
    
    global phylo
    
    phylo = phylo.set_index('index')
    phylo = phylo.loc[dataset.index]
    phylo = phylo.reset_index()
    
    dataset.apply(impute, axis=0)
    
    return dataset

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
    
    X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.25, shuffle = True, random_state=0)
    
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
    
    
    ## rm bad cols
    
    X_train = clean1(X_train)
    X_train = clean2(X_train)
    
    
    X_test = X_test[X_train.columns]

    full = pd.concat([X_train, X_test])

    global ibound

    ibound = len(X_train)

    full = rnan(full)

    X_train = full.iloc[:ibound]
    X_test = full.iloc[ibound:]


def train(dataset):
    
    global X_train
    global X_test
    global y_train
    global y_test
    global name

    ## fit model
    
    cachedir = mkdtemp()

    if args.dataset == 'summary':
        pipe = Pipeline([
            ('rf', RandomForestClassifier())
        ], memory = cachedir)
        param_grid = {
            'rf__n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
            'rf__max_features': ['sqrt', 'log2'],
            'rf__max_depth': [None, 3]
        }
    else:
        pipe = Pipeline([
            ('select', SelectFromModel(estimator=RandomForestClassifier(), threshold=-np.inf)),
            ('rf', RandomForestClassifier())
        ], memory = cachedir)
        param_grid = {
            'select__max_features': [int(x) for x in np.linspace(start = 20, stop = 500, num = 25)],
            'rf__n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
            'rf__max_features': ['sqrt', 'log2'],
            'rf__max_depth': [None, 3]
        }

    iters = 300

    if args.dataset == 'summary':
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
    if args.metric == 'prc':
        metrics.plot_precision_recall_curve(search, X_test, y_test)
        plt.savefig(name + '.png')
        plt.show()
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