#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import timeit
import sklearn.tree as tree
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, validation_curve, cross_val_score, learning_curve, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
import os

def hyperparameterOptimize(clf, X_train, y_train, parameter_grid):
    idx_random = np.random.choice(len(X_train) - 1, 5000)
    X_train_random = X_train.iloc[idx_random, :]
    y_train_random = y_train[idx_random]

    gsCV = GridSearchCV(clf, param_grid=parameter_grid, cv=4)
    gsCV.fit(X_train_random, y_train_random)
    print('score: {}'.format(gsCV.best_score_))
    print('params: {}'.format(gsCV.best_params_))

def genValidationCurve(clf, X_train, y_train, param_name, param_range, x_line):
    RandInd = np.random.choice(len(X_train), 5000)
    Rtrain = X_train.iloc[RandInd, :]
    Rlabels = y_train[RandInd]
    train_scores, test_scores = validation_curve(clf, Rtrain, Rlabels, param_name=param_name,
                                                 param_range=param_range, cv=4, scoring="accuracy", n_jobs=1)
    train_score_mean = np.mean(train_scores, axis=1)
    test_score_mean = np.mean(test_scores, axis=1)
    df_plot = pd.DataFrame(index=param_range,
                           data={"Training": 1 - train_score_mean,
                                 "Cross Validation": 1 - test_score_mean})
    ax = df_plot.plot(title='Validation Curve', fontsize=12)
    ax.set_xlabel(param_name)
    ax.set_ylabel('Error')
    # x_axis = data.index.get_level_values(0)
    ax.axvline(x=x_line, color='k')
    filename = "validate_" + type(clf).__name__ + "_" + str(X_train.shape[1]) + ".png"
    plt.savefig(filename)
    #plt.show()

def genLearningCurve(clf, X_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, train_sizes=np.array([0.1, 0.238, 0.4, 0.55, 0.65, 0.8, 1]), cv=5)
    train_score_mean = np.mean(train_scores, axis=1)
    test_score_mean = np.mean(test_scores, axis=1)
    df_plot = pd.DataFrame(index=train_sizes,
                           data={"Training": 1 - train_score_mean,
                                 "Cross Validation": 1 - test_score_mean})
    ax = df_plot.plot(title='Learning Curve', fontsize=12)
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Error')
    filename = "learn_" + type(clf).__name__ + "_" + str(X_train.shape[1]) + ".png"
    plt.savefig(filename)
    #plt.show()

def genConfusionMatrix(clf, X_train, X_test, y_train, y_test):
    # X_train = X_train2
    # X_test = X_test2
    # y_train = y_train2
    # y_test = y_test2
    # clf = clf1b

    start = timeit.default_timer()
    clf.fit(X_train, y_train)
    time_train = timeit.default_timer() - start

    start = timeit.default_timer()
    y_pred = clf.predict(X_test)
    time_test = timeit.default_timer() - start
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("train time " + str(round(time_test,4)))
    print("test time " + str(round(time_train,4)))
    print("accuracy " + str(round(accuracy,3)))
    print("f1 score " + str(round(f1,3)))

    stats_file = open("stats_" + type(clf).__name__ + "_" + str(X_train.shape[1]) + ".txt", "w")
    stats_file.write("train time " + str(round(time_train,4)))
    stats_file.write("\ntest time " + str(round(time_test,4)))
    stats_file.write("\naccuracy " + str(round(accuracy,3)))
    stats_file.write("\nf1 score " + str(round(f1,3)))
    stats_file.close()

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks,  ["0", "1"])
    plt.yticks(tick_marks,  ["0", "1"])
    normalize = False
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(2), range(2)):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    filename = "confuse_" + type(clf).__name__ + "_" + str(X_train.shape[1]) + ".png"
    plt.savefig(filename)
    #plt.show()

def getData(file):
    data = pd.read_csv(file)
    X = data.iloc[:, :-2]
    y = data.iloc[:, -2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
    features = list(X_train.columns.values)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return features, X_train, X_test, y_train, y_test



#####################################           READ IN DATA           #########################################
os.chdir('/Users/jeffwang/Dropbox/omscs/ml/P1Supervised_Learning')
features, X_train1, X_test1, y_train1, y_test1 = getData('phishing_clean.csv')
features2, X_train2, X_test2, y_train2, y_test2 = getData('winequality-data.csv')
y_train2 = (y_train2 >= 6).astype(int)
y_test2 = (y_test2 >= 6).astype(int)




######################################           Decision Tree          ########################################
#Phishing
clf1a = DecisionTreeClassifier(criterion ='entropy', max_depth =  15, min_samples_split =  2)
params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 4, 6, 10, 15, 30, 60, 100],
    'min_samples_split': [2, 5, 10, 15, 20, 25, 40, 50, 75, 100]
}
hyperparameterOptimize(clf1a, X_train1, y_train1, params)
#Best parameters: {'max_depth': 15, 'criterion': 'entropy', 'min_samples_split': 2}
validate_param = 'max_depth'
optimal_param = 15
genValidationCurve(clf1a, X_train1, y_train1, validate_param, params[validate_param], optimal_param)
genLearningCurve(clf1a, X_train1, y_train1)
genConfusionMatrix(clf1a, X_train1, X_test1, y_train1, y_test1)

#Coder
clf1b = DecisionTreeClassifier(criterion ='gini', max_depth = 30, min_samples_split = 2)
params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 4, 6, 10, 15, 30, 60, 100],
    'min_samples_split': [2, 5, 10, 15, 20, 25, 40, 50, 75, 100]
}
hyperparameterOptimize(clf1b, X_train2, y_train2, params)
#Best parameters: {'max_depth': 30, 'criterion': 'gini', 'min_samples_split': 2}
validate_param = 'max_depth'
optimal_param = 30
genValidationCurve(clf1b, X_train2, y_train2, validate_param, params[validate_param], optimal_param)
genLearningCurve(clf1b, X_train2, y_train2)
genConfusionMatrix(clf1b, X_train2, X_test2, y_train2, y_test2)




######################################           KNN           ########################################
#Phishing
clf2a = KNeighborsClassifier(n_neighbors = 15, weights ='distance', algorithm ='auto')
params = {
                 'n_neighbors' : [1, 2, 5, 10, 15, 50, 100,120, 130, 150],
                 'weights': ['uniform', 'distance'],
                 }
hyperparameterOptimize(clf2a, X_train1, y_train1, params)
#Best parameters: {'weights': 'distance', 'n_neighbors': 100}
validate_param = 'n_neighbors'
optimal_param = 15
genValidationCurve(clf2a, X_train1, y_train1, validate_param, params[validate_param], optimal_param)
genLearningCurve(clf2a, X_train1, y_train1)
genConfusionMatrix(clf2a, X_train1, X_test1, y_train1, y_test1)

#Coder
clf2b = KNeighborsClassifier(n_neighbors = 50, weights ='distance', algorithm ='auto')
params = {
                 'n_neighbors' : [1, 2, 5, 10, 15, 50, 100, 120, 150, 180],
                 'weights': ['uniform', 'distance'],
                 }
hyperparameterOptimize(clf2b, X_train2, y_train2, params)
#Best parameters: {'weights': 'distance', 'n_neighbors': 50}
validate_param = 'n_neighbors'
optimal_param = 50
genValidationCurve(clf2b, X_train2, y_train2, validate_param, params[validate_param], optimal_param)
genLearningCurve(clf2b, X_train2, y_train2)
genConfusionMatrix(clf2b, X_train2, X_test2, y_train2, y_test2)




######################################           Neural Networks           ########################################
#Phishing
clf3a = MLPClassifier(hidden_layer_sizes=(105,10,2),max_iter=1700)
clf3a.get_params()

params = {
                 'hidden_layer_sizes' : [(105,1,2),(105,2,2),(105,4,2),(105,6,2),(105,8,2),(105,10,2),(105,12,2)],
                 'max_iter': [2, 100, 200, 400, 600, 800, 1000, 1300, 1500, 1700, 2000]
                 }
hyperparameterOptimize(clf3a, X_train1, y_train1, params)
#Best parameters: {'hidden_layer_sizes': (105, 10, 2), 'max_iter': 1700}
validate_param = 'max_iter'
optimal_param = 1700
genValidationCurve(clf3a, X_train1, y_train1, validate_param, params[validate_param], optimal_param)
genLearningCurve(clf3a, X_train1, y_train1)
genConfusionMatrix(clf3a, X_train1, X_test1, y_train1, y_test1)

#Coder
clf3b = MLPClassifier(hidden_layer_sizes=(105,12,2),max_iter=1000)
params = {
                 'hidden_layer_sizes' : [(105,1,2),(105,2,2),(105,4,2),(105,6,2),(105,8,2),(105,10,2),(105,12,2)],
                 'max_iter': [2, 100, 200, 400, 600, 800, 1000, 1300, 1500, 1700, 2000]
                 }
hyperparameterOptimize(clf3b, X_train2, y_train2, params)
#Best parameters: {'hidden_layer_sizes': (105, 4, 2), 'max_iter': 2000}
validate_param = 'max_iter'
optimal_param = 2000
genValidationCurve(clf3b, X_train2, y_train2, validate_param, params[validate_param], optimal_param)
genLearningCurve(clf3b, X_train2, y_train2)
genConfusionMatrix(clf3b, X_train2, X_test2, y_train2, y_test2)




######################################           AdaBoost           ########################################
#Phishing
clf4a = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(criterion = 'entropy', max_depth =  5, min_samples_split = 100), learning_rate = 0.5, n_estimators=50)
params = {
                 'learning_rate' : [0.1,0.3,0.5,0.7,0.9,1],
                 'n_estimators': [1,10,20,30,40,50,70,100,200],
                 }
hyperparameterOptimize(clf4a, X_train1, y_train1, params)
#Best parameters: {'n_estimators': 50, 'learning_rate': 0.5}
validate_param = 'learning_rate'
optimal_param = .5
genValidationCurve(clf4a, X_train1, y_train1, validate_param, params[validate_param], optimal_param)
genLearningCurve(clf4a, X_train1, y_train1)
genConfusionMatrix(clf4a, X_train1, X_test1, y_train1, y_test1)

#Coder
clf4b = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(criterion = 'entropy', max_depth =  5, min_samples_split = 100), learning_rate = 0.3, n_estimators=200)
params = {
                 'learning_rate' : [0.1,0.3,0.5,0.7,0.9,1],
                 'n_estimators': [1,10,20,30,40,50,70,100,200],
                 }
hyperparameterOptimize(clf4b, X_train2, y_train2, params)
#Best parameters: {'n_estimators': 200, 'learning_rate': 0.3}
validate_param = 'learning_rate'
optimal_param = .3
genValidationCurve(clf4b, X_train2, y_train2, validate_param, params[validate_param], optimal_param)
genLearningCurve(clf4b, X_train2, y_train2)
genConfusionMatrix(clf4b, X_train2, X_test2, y_train2, y_test2)




######################################           SVC           ########################################
#Phishing
clf5a = SVC(C = 10, kernel = 'rbf', gamma = 0.1)
params = {
                 'gamma': [0.01,0.04,0.1,1,10,20,60,120],
                 'C': [0.01,0.1,1,10,100,300]
                 }
hyperparameterOptimize(clf5a, X_train1, y_train1, params)
#Best parameters: {'C': 10, 'gamma': 0.1}
validate_param = 'C'
optimal_param = 10
genValidationCurve(clf5a, X_train1, y_train1, validate_param, params[validate_param], optimal_param)
genLearningCurve(clf5a, X_train1, y_train1)
genConfusionMatrix(clf5a, X_train1, X_test1, y_train1, y_test1)

#Coder
clf5b = SVC(C = 100, kernel = 'rbf', gamma = 0.04)
params = {
                 'gamma': [0.01,0.04,0.1,1,10,20,60,120],
                 'C': [0.01,0.1,1,10,100,300]
                 }
hyperparameterOptimize(clf5b, X_train2, y_train2, params)
#Best parameters: {'C': 100, 'gamma': 0.04}
validate_param = 'C'
optimal_param = 100
genValidationCurve(clf5b, X_train2, y_train2, validate_param, params[validate_param], optimal_param)
genLearningCurve(clf5b, X_train2, y_train2)
genConfusionMatrix(clf5b, X_train2, X_test2, y_train2, y_test2)
