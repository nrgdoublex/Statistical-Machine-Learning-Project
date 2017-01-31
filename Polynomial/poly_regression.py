from __future__ import print_function
from __future__ import division

import numpy as np
import matrix_operation as mo
import input_data as id
import CV as cv
import matplotlib.pyplot as plt
import scipy.stats as stats

kfold = 20


validation_error = np.zeros((3,kfold))
testing_error = np.zeros((3,kfold))

for i in range(1,4):
    if i == 1:
        training_file = "training_data.txt"
        testing_file = "testing_data.txt"
    elif i == 2:
        training_file = "training_data_poly2.txt"
        testing_file = "testing_data_poly2.txt"
    elif i == 3:
        training_file = "training_data_poly3.txt"
        testing_file = "testing_data_poly3.txt"
        
    
    #training set for linear regression
    x,y,x_min,x_max,y_min,y_max = id.read_trainingdata(training_file)
    #testing set for linear regression
    x_test,y_test = id.read_testingdata(testing_file, x_min, x_max, y_min, y_max)
    x = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
    x_test = np.concatenate((np.ones((x_test.shape[0],1)),x_test),axis=1)
    
    #shuffle data before validation
    KFold_indices = cv.KFold(x,kfold)
    x_shuffle,y_shuffle = cv.shuffle(x,y)
    
    #cross validation
    for j in range(0,kfold):
        validation_data = x_shuffle[KFold_indices[j]:KFold_indices[j+1]]
        validation_target = y_shuffle[KFold_indices[j]:KFold_indices[j+1]]
        if j == 0:
            training_data = x_shuffle[KFold_indices[j+1]:]
            training_target = y_shuffle[KFold_indices[j+1]:]
        elif j == kfold - 1:
            training_data = x_shuffle[:KFold_indices[j]]
            training_target = y_shuffle[:KFold_indices[j]]
        else:
            training_data = np.append(x_shuffle[:KFold_indices[j]],x_shuffle[KFold_indices[j+1]:],axis=0)
            training_target = np.append(y_shuffle[:KFold_indices[j]],y_shuffle[KFold_indices[j+1]:],axis=0)
            
        #use closed form solution
        theta = np.dot(np.linalg.pinv(training_data),np.transpose(training_target))
        #diff
        valid_diff = np.subtract(validation_target,np.sum(np.multiply(validation_data,theta),axis=1))
        validation_error[i-1][j] = np.sqrt(np.inner(valid_diff,valid_diff)/valid_diff.shape[0])
        
        test_diff = np.subtract(y_test,np.sum(np.multiply(x_test,theta),axis=1))
        testing_error[i-1][j] = np.sqrt(np.inner(test_diff,test_diff)/test_diff.shape[0])
        
plt.figure()   
for i in range(0,3):
    linear_error = sorted(testing_error[i])
    fit_linear_error = stats.norm.pdf(linear_error, np.mean(linear_error), np.std(linear_error))
    plt.plot(linear_error,fit_linear_error,'-o',label='Degree-%d Polynomial' %(i+1))
plt.title('Validation Error of Polynomial Regression')
plt.xlabel('Mean Square Root Error')
plt.ylabel('Frequency')
plt.legend(loc="upper right")
plt.xscale('log')
plt.savefig("Testing_Error.png")

plt.figure()   
for i in range(0,3):
    linear_error = sorted(validation_error[i])
    fit_linear_error = stats.norm.pdf(linear_error, np.mean(linear_error), np.std(linear_error))
    plt.plot(linear_error,fit_linear_error,'-o',label='Degree-%d Polynomial' %(i+1))
plt.title('Validation Error of Polynomial Regression')
plt.xlabel('Mean Square Root Error')
plt.ylabel('Frequency')
plt.legend(loc="upper right")
plt.xscale('log')
plt.savefig("Validation_Error.png")

