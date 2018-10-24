#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 22:17:43 2018

@author: tron
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

data=pd.read_csv("turnover.csv")

print(data.head())

data.info()

# Print the unique values of the "department" column
print(data.department.unique())

# Print the unique values of the "salary" column
print(data.salary.unique())

#encode categories of the salary
data.salary=data.salary.astype('category')

data.salary=data.salary.cat.reorder_categories(['low', 'medium', 'high'])

data.salary=data.salary.cat.codes

#transforming department variable
departments=pd.get_dummies(data.department)
print(departments.head())

#droping columns and joining new dataframe to employee dataset
departments=departments.drop("accounting", axis=1)

data=data.drop("department", axis=1)

data=data.join(departments)


#percentage of employees who churn
n_employees=len(data)

print(data.churn.value_counts())

print(data.churn.value_counts()/n_employees*100)

#Separating Target and Features
target=data.churn
features=data.drop("churn", axis=1)

#Spliting employee data
target_train, target_test, features_train, features_test=train_test_split(target, features, test_size=0.25, random_state=42)

#Computing Gini index
stayed = 37
left = 1138

total=stayed+left

gini=2*(stayed/total)*(left/total)

#Splitting the tree
gini_A=0.65
gini_B=0.15

if gini_A < gini_B:
    print("split by A!")
else:
    print("split by B!")    

#Fitting the tree to employee data
model=DecisionTreeClassifier(random_state=42)
model.fit(features_train, target_train)
model.score(features_train, target_train)*100
model.score(features_test, target_test)*100    

#Exporting the tree
model.fit(features_train, target_train)
export_graphviz(model, "tree.dot")

#Pruning the tree
model_depth_5=DecisionTreeClassifier(max_depth=5, random_state=42)
print(model_depth_5.fit(features_train, target_train))
print(model_depth_5.score(features_train, target_train)*100)
print(model_depth_5.score(features_test, target_test)*100)

#Limiting the sample size
model_sample_100=DecisionTreeClassifier(min_samples_leaf=100, random_state=42)
model_sample_100.fit(features_train,target_train)
print(model_sample_100.score(features_train, target_train)*100)
print(model_sample_100.score(features_test, target_test)*100)

#Calculating accuracy metrics: precision
prediction=model.predict(features_test)
precision_score(target_test, prediction)

#Calculating accuracy metrics: recall
prediction=model.predict(features_test)
recall_score(target_test, prediction)

#Calculating the ROC/AUC score
prediction=model.predict(features_test)
roc_auc_score(target_test, prediction)

#Balancing classes
model_depth_5_b=DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
model_depth_5_b.fit(features_train, target_train)
print(model_depth_5_b.score(features_test, target_test)*100)

#Comparison of Employee attrition models: for balanced model
print(recall_score(target_test, prediction))
print(roc_auc_score(target_test, prediction))

model_depth_7_b=DecisionTreeClassifier(max_depth=7, class_weight="balanced", random_state=42)
model_depth_7_b.fit(features_train, target_train)
prediction_b=model_depth_7_b.predict(features_test)
print(recall_score(target_test, prediction_b))

#Cross-validation using sklearn
print(cross_val_score(model,features,target,cv=10))

#Setting up GridSearch parameters
depth=[i for i in range(5,21)]
samples=[i for i in range(50,500,50)]
parameters=dict(max_depth=depth, min_samples_leaf=samples)

#Implementing GridSearch
param_search=GridSearchCV(model, parameters)
param_search.fit(features_train, target_train)
print(param_search.best_params_)

#Sorting important features
feature_importances=model.feature_importances_
feature_list=list(features)
relative_importances=pd.DataFrame(index=feature_list, data=feature_importances, columns=["importance"])
relative_importances.sort_values(by="importance", ascending=False)
print(relative_importances.head())

#Selecting important features
selected_features=relative_importances[relative_importances.importance>0.01]
selected_list=selected_features.index
features_train_selected=features_train[selected_list]
features_test_selected=features_test[selected_list]

#Develop and test the best model
model_best=DecisionTreeClassifier(max_depth=8, min_samples_leaf=150, class_weight="balanced", random_state=42)
model_best.fit(features_train, target_train)
prediction_best=model_best.predict(features_test_selected)
print(model_best.score(features_test_selected, target_test)*100)
print(recall_score(prediction_best, target_test)*100)
print(roc_auc_score(prediction_best, target_test)*100)