import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

#Question1:
df_cancer = pd.DataFrame(data= np.c_[cancer['data']],
                         columns= cancer['feature_names'])
df_cancer['target'] = cancer['target']

#Question2:
cancerdf = df_cancer
    
dict = {0:'malignant', 1:'benign'}
cancerdf['target_label'] = cancerdf['target'].map(dict)

target = cancerdf.groupby(['target_label']).count()[['target']]
target = pd.Series(target['target'])

#Question3  
X = cancerdf[['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                'mean smoothness', 'mean compactness', 'mean concavity',
                'mean concave points', 'mean symmetry', 'mean fractal dimension',
                'radius error', 'texture error', 'perimeter error', 'area error',
                'smoothness error', 'compactness error', 'concavity error',
                'concave points error', 'symmetry error', 'fractal dimension error',
                'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                'worst smoothness', 'worst compactness', 'worst concavity',
                'worst concave points', 'worst symmetry', 'worst fractal dimension']]
y = cancerdf['target']
    
#Question4

from sklearn.model_selection import train_test_split
    
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Data Visualization
from matplotlib import cm

cmap = cm.get_cmap('gnuplot')
scatter = pd.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)


#Question5
from sklearn.neighbors import KNeighborsClassifier
    
knn = KNeighborsClassifier(n_neighbors = 1)

#Question6
means = cancerdf.mean()[:-1].values.reshape(1, -1)
knn.predict(means)

#Question7
cancer_prediction = []
knn.fit(X_train, y_train)

for items in range(len(X_test)):
    predict_test = knn.predict([X_test.iloc[items]])
    cancer_prediction.append(predict_test[0])
"""
#Question8
knn.score(X_test, y_test)



Function answer_one was answered correctly, 0.125 points were awarded.
['Warning, your solution for column target is of type int64, but the autograder is of type float64. Attempting to convert for grading.']
Function answer_two was answered incorrectly, 0.125 points were not awarded.
['Your solution shape (2, 1) was not the same as the autograder solution shape (2,).']
Function answer_three was answered correctly, 0.125 points were awarded.
['Warning, your solution is of type int64, but the autograder is of type float64. Attempting to convert for grading.']

Function answer_four was answered correctly, 0.125 points were awarded.
['Warning, your solution is of type int64, but the autograder is of type float64. Attempting to convert for grading.']
['Warning, your solution is of type int64, but the autograder is of type float64. Attempting to convert for grading.']

Function answer_five was answered incorrectly, 0.125 points were not awarded.
Function answer_six was answered incorrectly, 0.125 points were not awarded. The code for this problem may not have a return value.

Function answer_seven was answered incorrectly, 0.125 points were not awarded.
Student ndarray did not equal autograder ndarray.
Function answer_eight was answered correctly, 0.125 points were awarded.

"""