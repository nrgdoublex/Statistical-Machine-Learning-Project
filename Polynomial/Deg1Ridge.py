from __future__ import print_function
from __future__ import division

import numpy as np
import random
import math
import matrix_operation as mo
import matplotlib.pyplot as plt
import matplotlib.axes as axis
import input_data as id

#data
training_file = "training_data.txt"
testing_file = "testing_data.txt"

#training set
x,y,x_min,x_max,y_min,y_max = id.read_trainingdata(training_file)
#testing set
x_test,y_test = id.read_testingdata(testing_file, x_min, x_max, y_min, y_max)
#add constant feature
x = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
x_test = np.concatenate((np.ones((x_test.shape[0],1)),x_test),axis=1)

feature_dim = x.shape[1]

#parameters
learning_rate = 0.0001
regularization_set = [0.001,0.01,0.1,1,10,100]
precision = 0.0001
max_iteration = 1000
important_fea_thres = 0.1

#storing error
training_error = np.zeros((len(regularization_set),max_iteration))
testing_error = np.zeros((len(regularization_set),max_iteration))
ridge_coeff_idx = 0

#
for ridge_coeff in regularization_set:
    theta = np.random.rand(feature_dim) #constant feature
    gradient = np.zeros(feature_dim)
    iteration_idx = 0
    while iteration_idx < max_iteration:
        old_theta = theta
    #    if iteration_idx != 1 and np.mod(np.log10(iteration),1) == 0:
    #        learning_rate *= 0.5
        #get gradient and renew theta
        diff = np.dot(x,theta) - y
        gradient = (2) * (np.dot(np.transpose(x),diff) + ridge_coeff * theta)
        #print gradient
        theta = theta - learning_rate * gradient
        
        #training error
        diff = np.dot(x,theta) - y
        training_error[ridge_coeff_idx][iteration_idx] = np.sqrt(np.inner(diff,diff) / x.shape[0])
        diff = np.dot(x_test,theta) - y_test
        testing_error[ridge_coeff_idx][iteration_idx] = np.sqrt(np.inner(diff,diff) / x_test.shape[0])
    
        
        #check whether it's time to stop
        if np.linalg.norm(old_theta-theta,1) < precision:
            print("break")
            break
        
        iteration_idx += 1
        
    #count important features
    print(np.sum([np.abs(theta_i) > important_fea_thres for theta_i in theta]))
        
    ridge_coeff_idx += 1

#plot

for i in range(0,len(regularization_set)):
    plt.plot(np.arange(0,max_iteration),training_error[i],label='Regularization Coeff=%0.3f' %regularization_set[i])

plt.axes().set_xscale('log')
plt.title('Training Error of Ridge Regression with Degree 1')
plt.xlabel('Number of iterations')
plt.ylabel('Root Mean Square Error')
plt.legend(loc="upper right")
plt.savefig("Deg1TrainRidgeRegre.png")

plt.figure()
for i in range(0,len(regularization_set)):
    plt.plot(np.arange(0,max_iteration),testing_error[i],label='Regularization Coeff=%0.3f' %regularization_set[i])
plt.axes().set_xscale('log')
plt.title('Testing Error of Ridge Regression with Degree 1')
plt.xlabel('Number of iterations')
plt.ylabel('Root Mean Square Error')
plt.legend(loc="upper right")
plt.savefig("Deg1TestRidgeRegre.png")