import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

#Question1:
from sklearn.tree import DecisionTreeClassifier

clf2 = DecisionTreeClassifier(random_state=0).fit(X_train2, y_train2)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf2.score(X_train2, y_train2)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf2.score(X_test2, y_test2)))

Features_values = np.append((clf2.feature_importances_).reshape(-1,1),(np.array(X_train2.columns)).reshape(-1,1),axis=1)

Features_values = pd.DataFrame(Features_values) #convert to DF
Features_values = Features_values.sort_values(by=0, ascending = False).head(5) #sort dataframe and select top 5 features

top_5_imp_features = Features_values[1].tolist() #convert result to list

#array sort trial
#Features_values = Features_values[Features_values[:, ::-1].argsort()] #ref link https://thispointer.com/sorting-2d-numpy-array-by-column-or-row-in-python/
#Features_values = np.argsort(Features_values, axis=0)[:, ::-1]

#Question2
# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2

from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, random_state = 0)

clf = SVC(kernel = 'rbf', C = 1).fit(X_train, y_train)

print('Accuracy of rbf Kernerlised SVC classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of rbf Kernerlised SVC classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

#cross validation
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(clf, X_subset, y_subset,cv=3)
print('Cross-validation scores (3-fold):', cv_scores)
print('Mean cross-validation score (3-fold): {:.3f}'
     .format(np.mean(cv_scores)))

from sklearn.model_selection import validation_curve
param_range = np.logspace(-4, 1, 6)
train_scores, test_scores = validation_curve(SVC(), X_subset, y_subset,
                                            param_name='gamma',
                                            param_range=param_range, cv=3)


train_scores = np.mean(train_scores, axis = 1)
test_scores = np.mean(test_scores, axis = 1)

result_output_6 = [train_scores, test_scores]

#Question7

import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(np.logspace(-4,1,6), result_output_6[0], label='training data', markersize=10)
plt.plot(np.logspace(-4,1,6), result_output_6[1], label='test data', markersize=10)

plt.xlabel('Gamma')
plt.ylabel('Model Accuracy')
plt.legend(loc='best')
plt.show()
    
fit_assessment = [0.0001,10,0.001,0.01,0.1,1]
