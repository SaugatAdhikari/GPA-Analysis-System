__author__ = 'Saugat'


import pandas
import numpy as np
import itertools
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn import model_selection
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


url = "E:\\Study\\7th sem\\ML - COMP 484\\ML_files\\data.xls"
names = ['Stream','Semester','SLC_percent',
         'PlusTwo_percent', 'KUCAT', 'CGPA', 'Interactive level in lectures?','GPA Status', 'Seriousness in pre-exam breaks',
         'Daily hrs of study', 'Googling',
         'Personal_hobby','Do you do your Assignments yourself?', 'Attendance',
         'Interactive_in_lectures','Study_materials','Online_course',
         'Stay_KU']
dataset = pandas.read_excel(url, names=names)
# dataset2 = dataset['SLC_percent','PlusTwo_percent','KUCAT']

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

#Density Plot
dataset['SLC_percent'].plot(kind='density', subplots=True, layout=(1,1), sharex=False )
plt.show()

#
plt.xlabel('Students')
plt.ylabel('KUCAT Score')
plt.plot(dataset['KUCAT'],'r')
# fig = plt.figure()
# ax = fig.add_subplot(111)
ax.set_xlabel("2013", fontsize=12)

#histograms

dataset.hist()
plt.show()
#
#MULTIVARIATE PLOTS

scatter_matrix(dataset)
plt.show()

#Prediction on GPA
# dataset2.drop('CGPA', axis=1, inplace=True)
array = dataset.values
dataset2 = array[:,1:7]
cgpa=dataset['CGPA']
# print("CGPA")
# print(cgpa)
X, y = dataset2, cgpa
X_new = SelectKBest(f_regression, k="all").fit(X,y)
X_new1 = X_new.transform(X)#transform(X, y)
# print (X_new1.shape)
# print (X_new1)

# print(X.columns[X_new.get_support()])
cgpa=dataset['CGPA']
X_train, X_test,y_train, y_test=train_test_split(X_new1,cgpa,random_state=1)

print (X_test.shape)
print (X_train.shape)
print (y_train.shape)
print (y_test.shape)
print("  ")

linreg=linear_model.LinearRegression()
linreg.fit(X_train,y_train)

print(linreg.intercept_)
print("  ")
print(linreg.coef_)
print("  ")

y_pred=linreg.predict(X_test)
print (y_pred.shape)

print("Y_TEST")
print(y_test)
print("  ")

print("Y_PRED")
print(y_pred)
print("  ")

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


plt.xlabel('features')
plt.ylabel('test value, predicted value')
plt.plot(X_test,y_test,'.',X_test,y_pred,'-')
plt.show()
print("  ")

# linReg = LinearRegression()
# linReg.fit(X_train,y_train)
# prediction_linReg = linReg.predict(X_test)
# print("Accuracy Score Linear Model")
# print(accuracy_score(y_test, prediction_linReg))
# print("   ")


from sklearn import metrics
import numpy as np
print("Mean Absolute Error")
print (metrics.mean_absolute_error(y_test,y_pred))
print("  ")
print("Mean Squared Error")
print (metrics.mean_squared_error(y_test,y_pred))
print("  ")
print("Sqrt Mean Squared Error")
print (np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("  ")


#split-out validation dataset

array = dataset.values
X = array[:,1:7]
Y = array[:,7]
# print(X)
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

lr = LogisticRegression()
lda = LinearDiscriminantAnalysis()
knn = KNeighborsClassifier()
cart = DecisionTreeClassifier()
nb = GaussianNB()
svm = SVC()

lr.fit(X_train, Y_train)
lda.fit(X_train, Y_train)
knn.fit(X_train, Y_train)
cart.fit(X_train, Y_train)
nb.fit(X_train, Y_train)
svm.fit(X_train, Y_train)

predictions_lr = lr.predict(X_validation)
predictions_lda = lda.predict(X_validation)
predictions_knn = knn.predict(X_validation)
predictions = cart.predict(X_validation)
predictions_nb = nb.predict(X_validation)
predictions_svm = svm.predict(X_validation)

print("Accuracy Score LR")
print(accuracy_score(Y_validation, predictions_lr))
print("Accuracy Score LDA")
print(accuracy_score(Y_validation, predictions_lda))
print("Accuracy Score KNN")
print(accuracy_score(Y_validation, predictions_knn))
print("Accuracy Score CART")
print(accuracy_score(Y_validation, predictions))
print("Accuracy Score NB")
print(accuracy_score(Y_validation, predictions_nb))
print("Accuracy Score SVM")
print(accuracy_score(Y_validation, predictions_svm))


print("Confusion Matrix")
print(confusion_matrix(Y_validation, predictions))

print("Classification Report")
print(classification_report(Y_validation, predictions))

print(predictions)

plt.xlabel('features')
plt.ylabel('test value, predicted value')
plt.plot(X_validation,Y_validation,'.',X_validation,predictions,'-')
plt.show()
print("  ")
