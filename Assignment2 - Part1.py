import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
#def part1_scatter():
import matplotlib.pyplot as plt
plt.figure(figsize=(5,4))
plt.scatter(X_train, y_train, label='training data',marker= 'o')
#plt.scatter(X_test, y_test, label='test data')
plt.legend(loc=4);
plt.xlabel('Feature value (x)')
plt.ylabel('Target value (y)')
plt.show()

# NOTE: Uncomment the function below to visualize the data, but be sure 
# to **re-comment it before submitting this assignment to the autograder**.   
#part1_scatter()
    
#def answer_one():
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

result_output = np.empty((0,100), float)
# Your code here
for degree in [0, 1, 3, 6, 9]:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(x.reshape(-1,1)) #reshare x with 1D to 2D & transform into Poly function
    
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y,
                                                       random_state = 0)
    linreg = LinearRegression().fit(X_train, y_train)
         
    print('(poly deg {}) linear model coeff (w):\n{}'
         .format(degree,linreg.coef_))
    print('(poly deg {}) linear model intercept (b): {:.3f}'
         .format(degree,linreg.intercept_))
    print('(poly deg {}) R-squared score (training): {:.3f}'
         .format(degree, linreg.score(X_train, y_train)))
    print('(poly deg {}) R-squared score (test): {:.3f}\n'
         .format(degree, linreg.score(X_test, y_test))) 
    
    #prediction for 100 new inputs
    X_to_predict = np.linspace(0,10,100)
    X_to_predict_poly = poly.fit_transform(X_to_predict.reshape(-1,1)) #reshare x with 1D to 2D & transform into Poly function
        
    y_predict_output_poly = linreg.predict(X_to_predict_poly).reshape(1,-1) #reshare to transpose array as per assignment requirement
    
    result_output = np.append(result_output, np.array(y_predict_output_poly), axis=0)

#plot graph
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(X_train[:,[1]], y_train, 'o', label='training data', markersize=10)
plt.plot(X_test[:,[1]], y_test, 'o', label='test data', markersize=10)
for i,degree in enumerate([0,1,3,6,9]):
    plt.plot(np.linspace(0,10,100), result_output[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
plt.ylim(-1,2.5)
plt.legend(loc=4)
  
#Question2:
from sklearn.metrics.regression import r2_score
r2_train = np.empty((10,), float).flatten()
r2_test = np.empty((10,), float).flatten()

for i in range(10):
    poly = PolynomialFeatures(degree=i)
    X_poly = poly.fit_transform(x.reshape(-1,1)) #reshare x with 1D to 2D & transform into Poly function
    
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y,
                                                       random_state = 0)
    linreg = LinearRegression().fit(X_train, y_train)
             
    estimated_y_train = linreg.predict(X_train)
    r2_train_score = r2_score(y_train, estimated_y_train)
    r2_train[i] = np.array(r2_train_score)

    estimated_y_test = linreg.predict(X_test)
    r2_test_score = r2_score(y_test, estimated_y_test)
    r2_test[i] = np.array(r2_test_score)

    print('(poly deg {}) R-squared score (training): {:.3f}'
         .format(i, r2_score(y_train, estimated_y_train)))
    print('(poly deg {}) R-squared score (test): {:.3f}\n'
         .format(i, r2_score(y_test, estimated_y_test))) 

result_output_2 = np.array((r2_train,r2_test))

#Question3
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(np.linspace(0,9,10), r2_train, label='training data', markersize=10)
plt.plot(np.linspace(0,9,10), r2_test, label='test data', markersize=10)

plt.xlabel('Degree')
plt.ylabel('R-square score')
plt.legend(loc='best')
fit_assessment = [0,1,2,3,4,8,9,5,6,7]

#Question4
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics.regression import r2_score

poly = PolynomialFeatures(degree=12)
X_poly_12 = poly.fit_transform(x.reshape(-1,1)) #reshare x with 1D to 2D & transform into Poly function
    
X_train, X_test, y_train, y_test = train_test_split(X_poly_12, y,
                                                       random_state = 0)
linreg = LinearRegression().fit(X_train, y_train)

estimated_y_test_12 = linreg.predict(X_test)
r2_test_score_12 = r2_score(y_test, estimated_y_test_12)

print('(poly deg 12) R-squared score (test): {:.3f}\n'
         .format(r2_score(y_test, estimated_y_test_12))) 


linlasso = Lasso(alpha=0.01, max_iter = 10000).fit(X_train, y_train)

print('lasso regression linear model intercept: {}'
     .format(linlasso.intercept_))
print('lasso regression linear model coeff:\n{}'
     .format(linlasso.coef_))
print('Non-zero features: {}'
     .format(np.sum(linlasso.coef_ != 0)))
print('R-squared score (training): {:.3f}'
     .format(linlasso.score(X_train, y_train)))
print('R-squared score (test): {:.3f}\n'
     .format(linlasso.score(X_test, y_test)))

estimated_lasso_y_test_12 = linlasso.predict(X_test)
lasso_r2_test_score_12 = r2_score(y_test, estimated_lasso_y_test_12)

print('(poly deg 12) R-squared score (test): {:.3f}\n'
         .format(r2_score(y_test, estimated_lasso_y_test_12))) 

result_output_4 = [r2_test_score_12, lasso_r2_test_score_12]





