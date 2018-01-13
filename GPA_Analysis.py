__author__ = 'Saugat'


import pandas
import numpy as np
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


url = "E:\\Study\\7th sem\\ML - COMP 484\\ML_files\\data.xls"
names = ['Semester','Stream','SLC_percent',
         'PlusTwo_percent', 'KUCAT', 'CGPA', 'cleared_all?', 'Seriousness in pre-exam breaks',
         'Daily hrs of study', 'Googling',
         'Personal_hobby','Do you do your Assignments yourself?', 'Attendance',
         'Parents_profession','Interactive_in_lectures','Study_materials','Online_course',
         'Stay_KU','any_seniors?','Mentors?','family_problem','health_problem','Assignment_how?']
dataset = pandas.read_excel(url, names=names)

#shape

print("Data Shape")
print(dataset.shape)

#head

print(dataset.head(20))
# integer_dataset = dataset.loc(:,'SLC_percent':'CGPA')
# print(integer_dataset.head(10))
#
# #descriptions

print("Data Description")
print(dataset.describe())
print('')
#
# #class distribution

print("Grouping of Data")
print(dataset.groupby('Daily hrs of study').size())
print(' ')
print(dataset.groupby('Do you do your Assignments yourself?').size())

#Graph_CGPA

values = dataset.values
GPA = values[:,5]
plt.xlabel('Students')
plt.ylabel('CGPA Score')
plt.plot(GPA)
plt.show()
#


# #UNIVARIATE PLOTS
#box and whisker plots

dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()
#

#histograms

dataset.hist()
plt.show()
#
#MULTIVARIATE PLOTS

scatter_matrix(dataset)
plt.show()

#
#split-out validation dataset

array = dataset.values
X = array[:,1:7]
Y = array[:,7]
# print(Y)
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#test options and evaluation metric

seed = 7
scoring = 'accuracy'

#Spot Check Algorithms

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


#evaluate each model in turn

results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


#Compare Algorithms

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#make predictions on validation dataset

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print(predictions)
