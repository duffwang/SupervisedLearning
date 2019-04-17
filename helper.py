def plot_validation_curve(clf, X_train, y_train, param_name, param_range, x_line):
    RandInd = np.random.choice(len(X_train), 1000)
    Rtrain = X_train.iloc[RandInd, :]
    Rlabels = y_train[RandInd]
    train_scores, test_scores = validation_curve(clf, Rtrain, Rlabels, param_name=param_name,
                                                 param_range=param_range, cv=5, scoring="accuracy", n_jobs=1)
    train_score_mean = np.mean(train_scores, axis=1)
    test_score_mean = np.mean(test_scores, axis=1)
    df_plot = pd.DataFrame(index=param_range,
                           data={"Training": 1 - train_score_mean, "Cross Validation": 1 - test_score_mean})
    ax = df_plot.plot(title='Accuracy versus Number of Neighbors', fontsize=12)
    ax.set_xlabel('N of Neighbors')
    ax.set_ylabel('Error')
    # x_axis = data.index.get_level_values(0)
    ax.axvline(x=x_line, color='k')
    plt.show()

def plot_learning_curve(clf, X_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, train_sizes=np.array([0.1, 0.238, 0.4, 0.55, 0.65, 0.8, 1]), cv=5)
    train_score_mean = np.mean(train_scores, axis=1)
    test_score_mean = np.mean(test_scores, axis=1)

    df_plot = pd.DataFrame(index=train_sizes,
                           data={"Training": 1 - train_score_mean, "Cross Validation": 1 - test_score_mean})
    ax = df_plot.plot(title='Learning Curve', fontsize=12)
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Error')

    plt.show()

def plot_confusion_matrix(clf, X_train, X_test, y_train, y_test):
    start_time = timeit.default_timer()
    clf.fit(X_train, y_train)
    end_time = timeit.default_timer()
    training_time = end_time - start_time

    start_time = timeit.default_timer()
    y_pred = clf.predict(X_test)
    end_time = timeit.default_timer()
    pred_time = end_time - start_time

    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Model Evaluation Metrics Using Untouched Test Dataset")
    print("*****************************************************")
    print("Model Training Time (s):   " + "{:.5f}".format(training_time))
    print("Model Prediction Time (s): " + "{:.5f}\n".format(pred_time))
    print("F1 Score:  " + "{:.2f}".format(f1))
    print("Accuracy:  " + "{:.2f}".format(accuracy) + "     AUC:       " + "{:.2f}".format(auc))
    print("Precision: " + "{:.2f}".format(precision) + "     Recall:    " + "{:.2f}".format(recall))
    print("*****************************************************")
    plt.figure()
    classes = ["0", "1"]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    normalize = False
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(2), range(2)):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
