import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#Question1
fraud_data = pd.read_csv('fraud_data.csv')

count = fraud_data.groupby(['Class']).count()[['Amount']]
print(count)
percentage_fraud_transactions = count.loc[1]/count.loc[0]
percentage_fraud_transactions.loc['Amount']

X = fraud_data.iloc[:,:-1]
y = fraud_data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Question2
from sklearn.dummy import DummyClassifier
from sklearn.metrics import recall_score

dummy_most_freq = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)

Accuracy_test = dummy_most_freq.score(X_test, y_test)

y_dummy_predictions = dummy_most_freq.predict(X_test)

Recall = recall_score(y_test, y_dummy_predictions)

acc_recall_tuple = (Accuracy_test,Recall)
print(acc_recall_tuple)

confusion_matrix(y_test, y_dummy_predictions)

#Question3
from sklearn.metrics import recall_score, precision_score
from sklearn.svm import SVC

clf = SVC().fit(X_train,y_train)
accuracy = clf.score(X_test, y_test)
    
y_clf_predict = clf.predict(X_test)

recall = recall_score(y_test,y_clf_predict)
precision = precision_score(y_test,y_clf_predict)
    
acc_recall_prec_tuple = (accuracy,recall,precision)
print(acc_recall_prec_tuple)


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_clf_predict) #test data

#Question4

clf = SVC(C= 1e9, gamma = 1e-07).fit(X_train,y_train)

y_predicted_dec_func = clf.decision_function(X_test) > -220

array_q4 = confusion_matrix(y_test, y_predicted_dec_func)

#Question5
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

lr = LogisticRegression().fit(X_train, y_train)

y_proba_lr = lr.predict_proba(X_test)

precision, recall, thresholds = precision_recall_curve(y_test, y_proba_lr[:,1])

plt.figure()
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.plot(precision, recall, label='Precision-Recall Curve')
plt.xlabel('Precision', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.axes().set_aspect('equal')
plt.show()

Precision_index = np.where(precision == 0.75)
recall_075 = recall[Precision_index]


fpr_lr, tpr_lr, thresholds = roc_curve(y_test, y_proba_lr[:,1])
roc_auc_lr = auc(fpr_lr, tpr_lr)

fpr_index = np.nonzero(np.around(fpr_lr, decimals = 2)== 0.16)[0][0]

tpr_016 = tpr_lr[fpr_index]

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.axes().set_aspect('equal')
plt.show()

#Question5
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression().fit(X_train, y_train)
grid_values = {'penalty': ['l1','l2'],
               'C':[0.01, 0.1, 1, 10, 100]}

grid_lr_recall = GridSearchCV(lr, param_grid = grid_values,scoring = 'recall', cv=3)
grid_lr_recall.fit(X_train, y_train)
#y_decision_fn_scores_recall = grid_lr_recall.decision_function(X_test)

print('Grid best parameter (max. recall): ', grid_lr_recall.best_params_)
print('Grid best score (recall): ', grid_lr_recall.best_score_)
print('Grid mean test score (recall): ', grid_lr_recall.cv_results_['mean_test_score'])

print('Grid params (recall): ', grid_lr_recall.cv_results_['params'])
all_l1 = grid_lr_recall.cv_results_['mean_test_score'][::2]

all_l2 = grid_lr_recall.cv_results_['mean_test_score'][1::2] 


np.column_stack((all_l1,all_l2))


