# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 21:43:08 2020

@author: anukriti
"""

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import pickle
import sklearn.ensemble as ske
from sklearn import tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from tpot import TPOTClassifier

data = pd.read_csv('data.csv', sep='|')
data.legitimate.value_counts()
X = data.drop(['Name', 'md5', 'legitimate'], axis=1).values
y = data['legitimate'].values

# Feature selection using Trees Classifier 
fsel = ske.ExtraTreesClassifier().fit(X, y)
model = SelectFromModel(fsel, prefit=True)
X_new = model.transform(X)
nb_features = X_new.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X_new, y ,test_size=0.2) 

### Genetic Algorithm --> TPOT Classifier 
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt','log2']
max_depth = [int(x) for x in np.linspace(10, 1000,10)]
min_samples_split = [2, 5, 10,14]
min_samples_leaf = [1, 2, 4,6,8]

param = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}

tpot_classifier = TPOTClassifier(generations= 5, population_size= 24, offspring_size= 12,
                                 verbosity= 2, early_stop= 12, 
                                 config_dict={'sklearn.ensemble.RandomForestClassifier': param}, 
                                 cv = 4, scoring = 'accuracy')
tpot_classifier.fit(X_train,y_train)

## Evaluation (Genetic Algorithm) ------------> (7)
accuracy = tpot_classifier.score(X_test, y_test)
print(accuracy)
'''
predictionforest = tpot_classifier.predict(X_test)
print(confusion_matrix(y_test,predictionforest))
print(accuracy_score(y_test,predictionforest))
print(classification_report(y_test,predictionforest)) 
acc5 = accuracy_score(y_test,predictionforest)
'''