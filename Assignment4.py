
#Prediction objective - 
#to predict whether a given blight ticket will be paid on time
#Predict target variable 'compliance' as 
#1 - will be paid on time or 
#0 - will not be paid on time

import pandas as pd
import numpy as np
#Stage 1 - Read and cleaning Training data set
addresses = pd.read_csv('addresses.csv',engine='python')
latlons = pd.read_csv('latlons.csv',engine='python')
add_latlon = pd.merge(addresses,latlons, how ='left', on=['address'])
add_latlon.drop(['address'], axis=1, inplace=True)

train_raw = pd.read_csv('train.csv',engine='python')
train = train_raw.copy()


train = pd.merge(train,add_latlon, how ='left', on=['ticket_id'])

train.dropna(subset = ["compliance"], inplace=True) #compliance = NaN, violation not responsible to pay fine

#drop columns to avoid data leakage
train.drop(['payment_amount','payment_date','payment_status','balance_due','collection_status','compliance_detail']
           , axis=1, inplace=True)

#drop useless columns
train.drop(['ticket_id','non_us_str_code','violation_zip_code','grafitti_status','mailing_address_str_number'], axis=1, inplace=True)
#Drop all remaining rows with NaN
train = train.dropna()

#Adding feature - no of days between ticket issue date & hearing date
train[['ticket_issued_date','hearing_date']] = train[['ticket_issued_date','hearing_date']].apply(pd.to_datetime)
train['Issue_To_hearing_days'] = (train['hearing_date'] - train['ticket_issued_date']).dt.days


#Further drop in columns - date columns and granular fee columns
train.drop(['ticket_issued_date','hearing_date','fine_amount','admin_fee','state_fee','late_fee','discount_amount','clean_up_cost']
           , axis=1, inplace=True)

#Further drop in columns - not significant differnec in info and repeate info

train.drop(['violator_name','country','violation_description','violation_street_number','mailing_address_str_name']
           , axis=1, inplace=True)   

#Dropped features to explore in future
train.drop(['agency_name','inspector_name','violation_street_name','city','state', 'zip_code','disposition','violation_code']
           , axis=1, inplace=True)

#engineer 'violation_code' feature
train['violation_code'] = np.where(train['violation_code'].isin(['9-1-36(a)',
'9-1-81(a)','22-2-88','9-1-104','22-2-88(b)','22-2-45','9-1-43(a) - (Dwellin',
'9-1-105','9-1-110(a)','22-2-22','9-1-103(C)','19450901','22-2-43','22-2-17','22-2-61']), train['violation_code'], 'Other')

#Converting Ordinal columns to numerical value
nominal_columns = ['agency_name', 'disposition','violation_code']
dummy_df = pd.get_dummies(train[nominal_columns])
train = pd.concat([train, dummy_df], axis=1)
train = train.drop(nominal_columns, axis=1)

y = train['compliance'] #This should never be dataframe
X = train.drop(['compliance']
           , axis=1, inplace=False) 

"""
#Stage 2 - Synchronize test set data
test_raw = pd.read_csv('test.csv',encoding = "ISO-8859-1",low_memory=False)
test = test_raw.copy()

test = pd.merge(test,add_latlon, how ='left', on=['ticket_id'])

#drop useless columns
test_raw = pd.read_csv('test.csv',encoding = "ISO-8859-1",low_memory=False)
test = test_raw.copy()
test = pd.merge(test,add_latlon, how ='left', on=['ticket_id'])
    
#drop useless columns
test.drop(['ticket_id','non_us_str_code','violation_zip_code','grafitti_status','mailing_address_str_number'], axis=1, inplace=True)

#Adding feature - no of days between ticket issue date & hearing date
test[['ticket_issued_date','hearing_date']] = test[['ticket_issued_date','hearing_date']].apply(pd.to_datetime)
test['Issue_To_hearing_days'] = (test['hearing_date'] - test['ticket_issued_date']).dt.days

#Further drop in columns - date columns and granular fee columns
test.drop(['ticket_issued_date','hearing_date','fine_amount','admin_fee','state_fee','late_fee','discount_amount','clean_up_cost']
               , axis=1, inplace=True)

#Further drop in columns - not significant differnec in info and repeate info

test.drop(['violator_name','country','violation_description','violation_street_number','mailing_address_str_name']
               , axis=1, inplace=True)   

#Dropped features to explore in future
test.drop(['agency_name','inspector_name','violation_street_name','city','state', 'zip_code','disposition','violation_code']
                , axis=1, inplace=True) 

#replace nan values by zero
test['Issue_To_hearing_days'] = test['Issue_To_hearing_days'].fillna(0)
test['lat'] = test['lat'].fillna(0)
test['lon'] = test['lon'].fillna(0)

#engineer 'violation_code' feature
test['violation_code'] = np.where(test['violation_code'].isin(['9-1-36(a)',
'9-1-81(a)','22-2-88','9-1-104','22-2-88(b)','22-2-45','9-1-43(a) - (Dwellin',
'9-1-105','9-1-110(a)','22-2-22','9-1-103(C)','19450901','22-2-43','22-2-17','22-2-61']), test['violation_code'], 'Other')

nominal_columns = ['agency_name', 'disposition','violation_code']
dummy_test_df = pd.get_dummies(test[nominal_columns])
test = pd.concat([test, dummy_test_df], axis=1)
test = test.drop(nominal_columns, axis=1)

#Drop extra columns
test.drop(['disposition_Responsible by Dismissal','disposition_Responsible - Compl/Adj by Default',
           'disposition_Responsible - Compl/Adj by Determi','disposition_Responsible (Fine Waived) by Admis']
          , axis=1, inplace=True)

#add columns with zero values
test['agency_name_Health Department'] = 0
test['agency_name_Neighborhood City Halls'] = 0

#replace nan values by zero
test['Issue_To_hearing_days'] = test['Issue_To_hearing_days'].fillna(0)
"""
#Satge 3 - Feature Engineering - Analysis utilities
#Data Visualization
train_raw.columns.tolist()

null_counts = train.isnull().sum()
#print("Number of null values in each column:\n{}".format(null_counts))

#https://www.dataquest.io/blog/machine-learning-preparing-data/
#one hot encoder - https://towardsdatascience.com/choosing-the-right-encoding-method-label-vs-onehot-encoder-a4434493149b
#convering categorical features to numerival


#Logic to find unique values within each feature with 'Object' category
print("Data types and their frequency\n{}".format(train.dtypes.value_counts()))

object_columns_df = train_raw.select_dtypes(include=['object'])

cols = ['violation_code']
for name in cols:
    print(name,':')
    print(object_columns_df[name].value_counts(),'\n')

train_raw['violation_street_name'].value_counts() 


from matplotlib import cm
cmap = cm.get_cmap('gnuplot')
scatter = pd.plotting.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)


#Stage 4 - Analysis Utilities
#To check data imbalance
train_raw['compliance'].value_counts() 

#Stage 5 - Model training

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc #gives same result as roc_auc_score
from sklearn.metrics import roc_auc_score #Used in grid search, use best parameters given by grid search
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Negative class (0) is most frequent
#Dummy classifier
clf = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
# Therefore the dummy 'most_frequent' classifier always predicts class 0

#Score to beat - 0.928

#KNN classifier - decision function not supported
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors = 5).fit(X_train, y_train)

#Logistic regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1, solver='lbfgs').fit(X_train, y_train)

#Decision tree classifier - decision function not 
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_decision_tree

clf = DecisionTreeClassifier(max_depth = 5).fit(X_train, y_train)

#Visualizing decision trees
plot_decision_tree(clf, X.columns, ['0','1'])


#Support Vector machines Classifier
from sklearn.svm import SVC
clf = SVC(kernel = 'rbf', gamma = 100).fit(X_train, y_train)

#Random forestor
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 1000, max_features = 1, random_state = 0) #random_state - to control inherant randomness in generating bootstrap samples to build trees
clf.fit(X_train, y_train)

#Gradient boosted
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(learning_rate = .01, max_depth = 3, n_estimators=1000, random_state = 0)
clf.fit(X_train, y_train)

#Stage 6- Model training evaluation
print('Accuracy of classifier on training set: {:.4f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of classifier on test set: {:.4f}'
     .format(clf.score(X_test, y_test)))
print('Feature importances: {}'.format(clf.feature_importances_))

#confusion Metric
y_predicted = clf.predict(X_test)
confusion_matrix(y_test, y_predicted)

print('Precision: {:.2f}'.format(precision_score(y_test, y_predicted,average= 'binary')))
print('Recall: {:.2f}'.format(recall_score(y_test, y_predicted,average= 'binary')))
print('F1: {:.2f}'.format(f1_score(y_test, y_predicted,average= 'binary')))

#Calculate ROC curve
y_deci_func_clf = clf.fit(X_train, y_train).decision_function(X_test)
fpr_lr, tpr_lr, thresholds = roc_curve(y_test, y_deci_func_clf)

fpr_lr, tpr_lr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1]) #if decision function is not available
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_lr, tpr_lr, lw=3, label='ROC curve for given model (area = {:0.2f})'.format(roc_auc_lr))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve for given model', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.axes().set_aspect('equal')
plt.show()

roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])


#Predit test set classification probability
y_proba_func_clf = clf.fit(X_train, y_train).predict_proba(X_test)

y_proba_list = list(zip(y_test[0:40], y_proba_func_clf[0:40,1]))

# show the probability of compliance class for first 40 instances
y_proba_list

#Probability for test_unseen data
y_proba_func_clf_test_unseen = clf.fit(X_train, y_train).predict_proba(test)
compliance_probability = pd.Series(y_proba_func_clf_test_unseen[:,1], index= test_raw['ticket_id'].tolist())

#Grid Search to optomize AUC

clf = SVC(kernel='rbf')
grid_values = {'gamma': [10,100]}

# alternative metric to optimize over grid parameters: AUC
grid_clf_auc = GridSearchCV(clf, param_grid = grid_values, scoring = 'roc_auc',cv=3)
grid_clf_auc.fit(X_train, y_train)
y_decision_fn_auc_best_param = grid_clf_auc.decision_function(X_test)

print('Test set AUC: ', roc_auc_score(y_test, y_decision_fn_auc_best_param))
print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
print('Train set Grid best score (AUC): ', grid_clf_auc.best_score_)

#Map plotting
import matplotlib.pyplot as plt

BBox = ((train.lon.min(),   train.lon.max(),      
         train.lat.min(), train.lat.max()))
        
fig, ax = plt.subplots(figsize = (8,7))
ax.scatter(train.lon, train.lat, zorder=1, alpha= 0.2, c='b', s=10)
ax.set_title('Plotting Spatial Data of Detroitte')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])

ax.imshow(zorder=0, extent = BBox, aspect= 'equal')


