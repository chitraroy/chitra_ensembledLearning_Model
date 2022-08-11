# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 21:53:07 2022

@author: chitr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Load & check the data:    


df_chitra = pd.read_csv('C:/Users/chitr/OneDrive/Desktop/Machine_learning/Ensembled Learning/pima-indians-diabetes.csv',names=['preg','plas','pres','skin','test','mass','pedi','age','class'])
df_chitra.head()

df_chitra.dtypes

df_chitra.select_dtypes(include=['category']).dtypes

df_chitra.columns

df_chitra.isnull().sum()

df_chitra.describe()



import seaborn as sns
print(df_chitra['class'].value_counts())
sns.countplot(x='class', data=df_chitra)


#  Pre-process and prepare the data for machine learning    

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

transformer_chitra = ColumnTransformer([("scale", StandardScaler() , [x for x in range(0,8)])])

X = df_chitra.drop(columns=['class']).values
Y = df_chitra['class'].values

X = transformer_chitra.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train_chitra , X_test_chitra, Y_train_chitra, Y_test_chitra = train_test_split(X, Y, test_size=0.30, random_state=42)

# Declearing classifiers 

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

logistics_regression_C = LogisticRegression(max_iter=1400)
random_forest_C = RandomForestClassifier()
svm_C = SVC()
decision_tree_C = DecisionTreeClassifier(criterion="entropy",max_depth = 42)
extra_trees_C = ExtraTreesClassifier()

# Hard Voting Instances

from sklearn.ensemble import VotingClassifier
voting_classifier_chitraHard = VotingClassifier(estimators=[('LR', logistics_regression_C), 
                                                       ('RF', random_forest_C), 
                                                       ('SVM', svm_C), 
                                                       ('DT',decision_tree_C),
                                                       ('ET',extra_trees_C)], 
                                           voting='hard')


voting_classifier_chitraHard.fit(X_train_chitra, Y_train_chitra)

voting_classifier_chitraHard.predict(X_test_chitra[:3])


Y_test_chitra[:3]

estimators = ['DT', 'ET', 'LR', 'RF', 'SVM']

ind = 1
for instance in X_test_chitra[:3]:
  print("Instance #" ,ind)
  ind+=1
  for estimator in estimators:
    vote = voting_classifier_chitraHard.named_estimators_[estimator].predict(instance.reshape(1, -1))
    print('Model =' , estimator, ', Vote =', vote)
    
    


# Soft Voting Instances

svm_C = SVC(probability=True)

voting_classifier_ChitraSoft = VotingClassifier(estimators=[('LR', logistics_regression_C), 
                                                       ('RF', random_forest_C), 
                                                       ('SVM', svm_C), 
                                                       ('DT',decision_tree_C),
                                                       ('ET',extra_trees_C)], 
                                           voting='soft')


voting_classifier_ChitraSoft.fit(X_train_chitra, Y_train_chitra)

voting_classifier_ChitraSoft.predict(X_test_chitra[:3])


Y_test_chitra[:3]



ind = 1
for instance in X_test_chitra[:3]:
  print("Instance #" ,ind)
  ind+=1
  for estimator in estimators:
    vote = voting_classifier_ChitraSoft.named_estimators_[estimator].predict(instance.reshape(1, -1))
    print('Model =' , estimator, ', Vote =', vote)


# compareing Decision tree classifier and Extra tree classifier


from sklearn.pipeline import Pipeline

pipeline1_chitra = Pipeline([('scaler', transformer_chitra),
                             ('et', extra_trees_C)])
pipeline2_chitra = Pipeline([('scaler', transformer_chitra),
                             ('DT', decision_tree_C)])

pipeline1_chitra.fit(X_train_chitra, Y_train_chitra)
pipeline2_chitra.fit(X_train_chitra, Y_train_chitra)


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(n_splits=10, random_state=42, shuffle=True)
scores = cross_val_score(pipeline1_chitra, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
scores = cross_val_score(pipeline2_chitra, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


from sklearn.metrics import confusion_matrix, classification_report

y_pred1 = pipeline1_chitra.predict(X_test_chitra)
print(confusion_matrix(y_pred1, Y_test_chitra))


print(classification_report(y_pred1, Y_test_chitra))


y_pred2 = pipeline2_chitra.predict(X_test_chitra)
print(confusion_matrix(y_pred2, Y_test_chitra))

print(classification_report(y_pred2, Y_test_chitra))



# Fine tuning the model with Randomized search in extra tree classifier




from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

param_dist = {"et__n_estimators": randint(10,3000,20),
              "et__max_depth": randint(1,1000,2)}

extra_trees_74 = RandomizedSearchCV(pipeline1_chitra, param_dist, cv=5)
extra_trees_74.fit(X_train_chitra,Y_train_chitra)

extra_trees_74.best_params_

fine_tunes_chitra = extra_trees_74.best_estimator_
fine_tunes_chitra
y_tunedPred = fine_tunes_chitra.predict(X_test_chitra)

scores = cross_val_score(fine_tunes_chitra, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

print(confusion_matrix(y_tunedPred, Y_test_chitra))

print(classification_report(y_tunedPred, Y_test_chitra))

