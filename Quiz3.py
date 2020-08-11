
#Q5
y_scores_lr = m.fit(X_train, y_train).decision_function(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, y_scores_lr)

print(precision)
print(recall)

#Q8
svm_predicted_mc = m.predict(X_test)
print('Macro-averaged precision = {:.2f} (treat classes equally)'
      .format(precision_score(y_test, svm_predicted_mc, average = 'macro')))

#Q13
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.svm import SVC
grid_values = {'gamma': [0.01, 0.1, 1, 10],
               'C':[0.01, 0.1, 1, 10]}
grid_clf_rec = GridSearchCV(m, param_grid = grid_values,scoring = 'recall',cv=3)
grid_clf_rec.fit(X_train, y_train)
print('Train Best parameters-Recall=',grid_clf_rec.best_params_)
print('Train Best score-Recall={:.3f}'.format(grid_clf_rec.best_score_))

y_pred = grid_clf_rec.best_estimator_.predict(X_test)

precision = precision_score(y_test, y_pred)
print('Precision for Best Recall score={:.3f}'.format(precision))

#Q14
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.svm import SVC
grid_values = {'gamma': [0.01, 0.1, 1, 10],
               'C':[0.01, 0.1, 1, 10]}
grid_clf_rec = GridSearchCV(m, param_grid = grid_values,scoring='precision',cv=3)
grid_clf_rec.fit(X_train, y_train)
print('Train Best parameters-Precision=',grid_clf_rec.best_params_)
print('Train Best score-Precision={:.3f}'.format(grid_clf_rec.best_score_))

y_pred = grid_clf_rec.best_estimator_.predict(X_test)

recall = recall_score(y_test, y_pred)
print('Recall for Best Precision score={:.3f}'.format(recall))


#self study - getting under the hood - Multiclass classification

# 1 Data gathering - Digit Dataset
from sklearn.datasets import load_digits

dataset = load_digits()
X, y = dataset.data, dataset.target

# 2 Multiclass SVC classifier with RBF kernel

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# The default SVC kernel is radial basis function (RBF)
gamma = [0.00001,0.0001,0.01, 0.1, 1, 10]
c = [0.01, 0.1, 1, 10]
for this_C in c:
    clf = SVC(kernel = 'rbf', gamma = .0001, C = this_C).fit(X_train, y_train)

    print('Accuracy of RBF-kernel SVC on training set: {:.2} for C:{:.2f}'
          .format(clf.score(X_train, y_train),this_C))
    print('Accuracy of RBF-kernel SVC on test set: {:.2f} for C:{:.2f}'
          .format(clf.score(X_test, y_test),this_C))

# conclusions - gamma = .0001 and C = 10 optized parameters
#accuracy - tarinig set = 1
#accuracy = test set = 0.98

# 3 cross validation
#Multiclass format not supported
from sklearn.model_selection import cross_val_score
clf = SVC(kernel = 'rbf', gamma = .0001, C = 10).fit(X_train, y_train)
print('Accuracy of RBF-kernel SVC on training set: {:.2}'
          .format(clf.score(X_train, y_train)))
print('Accuracy of RBF-kernel SVC on test set: {:.2f}'
          .format(clf.score(X_test, y_test)))


cv_scores = cross_val_score(clf, X_train, y_train,cv=2) #this works as well
print('Cross-validation scores (3-fold):', cv_scores)
print('Mean cross-validation score (3-fold): {:.3f}'
     .format(np.mean(cv_scores)))

# 4 Validation curve - This is similar to grid search with one parameter
from sklearn.model_selection import validation_curve
train_scores, test_scores = validation_curve(SVC(C = 10), X_train, y_train,
                                            param_name='gamma',
                                            param_range=[0.00001,0.0001,0.01], 
                                            cv=5)

print(train_scores)
print(test_scores)

# 5 Dummy classifier
from sklearn.dummy import DummyClassifier
dummy_majority = DummyClassifier(strategy = 'uniform').fit(X_train, y_train)
y_dummy_predictions = dummy_majority.predict(X_test)
dummy_majority.score(X_test, y_test)

from collections import Counter 
list(zip(Counter(y_train).keys(), Counter(y_train).values())) #to find most frequest class

# 6 Multiclass confusion Metric
from sklearn.metrics import confusion_matrix

clf_predicted_test = clf.predict(X_test)
confusion_matrix(y_test, clf_predicted_test) #test data

clf_predicted_train = clf.predict(X_train)
confusion_matrix(y_train, clf_predicted_train) #train data

# 7 Metric report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, clf_predicted_test))) 
print('Score: {:.2f}'.format(clf.score(X_test, y_test))) #Accuracy & Score gives same result

print('Precision: {:.2f}'.format(precision_score(y_test, clf_predicted_test,average = 'macro')))
print('Recall: {:.2f}'.format(recall_score(y_test, clf_predicted_test,average = 'macro')))
print('F1: {:.2f}'.format(f1_score(y_test, clf_predicted_test,average = 'macro')))# all metrics

from sklearn.metrics import classification_report
print('SVM\n', 
      classification_report(y_test, clf_predicted_test))

# 8 decision Function
y_scores_dec_func = clf.fit(X_train, y_train).decision_function(X_test)


# 9 Recall precision curve
from sklearn.metrics import precision_recall_curve
#Multiclass format not supported

# 10 Roc curve and AUC
from sklearn.metrics import roc_curve, auc
#Multiclass format not supported

# 11 Cross validation score for 'roc_auc', 'precison','recall'
from sklearn.model_selection import cross_val_score

# 12 Grid Search
from sklearn.model_selection import GridSearchCV

clf = SVC(kernel = 'rbf').fit(X_train, y_train)
grid_values = {'gamma': [0.00001,0.0001,0.01, 0.1, 1, 10],
               'C':[0.01, 0.1, 1, 10]}
# default metric to optimize over grid parameters: accuracy
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values,cv=3)
grid_clf_acc.fit(X_train, y_train)

print('Grid best parameter (max. accuracy): ', grid_clf_acc.best_params_)
print('Grid best score (accuracy): ', grid_clf_acc.best_score_)

