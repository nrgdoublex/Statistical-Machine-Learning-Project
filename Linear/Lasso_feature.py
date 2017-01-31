import numpy as np
import random
import math
import matrix_operation as mo
import matplotlib.pyplot as plt
import matplotlib.axes as axis

#parameters
feature_dim = 16

#generate matrix
training = open("training_data.txt")
training_matrix = []
lines = training.readlines()
for line in lines:
    list = line.rstrip('\n').split(',')
    list = map(float,list)
    training_matrix.append(list)
training.close()
#normalize data-both x and y
x = np.array(mo.col_submatrix(training_matrix,0,feature_dim))
y = np.array(mo.getcolumnvector(training_matrix,feature_dim))
x_min = np.min(x, axis=0)
x_max = np.max(x, axis=0)
x = (x - x_min) / (x_max - x_min)
y_min = np.min(y)
y_max = np.max(y)
y = (y - y_min) / (y_max - y_min)
x = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)  
 
#testing
testing = open("testing_data.txt")
testing_matrix = []
lines = testing.readlines()
for line in lines:
    list = line.rstrip('\n').split(',')
    list = map(float,list)
    testing_matrix.append(list)
testing.close()
x_test = np.array(mo.col_submatrix(testing_matrix,0,feature_dim))
y_test = np.array(mo.getcolumnvector(testing_matrix,feature_dim))
x_test = (x_test - x_min) / (x_max - x_min)
y_test = (y_test - y_min) / (y_max - y_min)
x_test = np.concatenate((np.ones((x_test.shape[0],1)),x_test),axis=1) 
    
#parameters
learning_rate = 0.0001 #around 0.0001 level
precision = 0.0001
max_iteration = 1000
#Lasso regularization constant
lasso = 0.001
#count important features
feature_selected_count = np.zeros(feature_dim+1)
#k largest features
num_largest_feature = 5

num_exp = 1000
#count nonzero theta
#nonzero_theta = np.zeros(max_iteration)
        

#main stochastic gradient descent algorithm
sample_dim = x.shape[0]
for num_exp_idx in range(0,num_exp):
    theta = np.random.rand(feature_dim+1) #constant feature
    gradient = np.zeros(feature_dim+1)
    iteration_idx = 0
    training_error = np.zeros(max_iteration)
    testing_error = np.zeros(max_iteration)
    while iteration_idx < max_iteration:
        old_theta = theta
        
        #get gradient and renew theta
        diff = np.dot(x,theta) - y
        gradient = (2) * np.dot(np.transpose(x),diff)
        theta = theta - learning_rate * gradient
        #soft-thresholding
        for i in range(0,len(theta)):
            if theta[i] > lasso:
                theta[i] = theta[i] - lasso
            elif theta[i] < (-1) * lasso:
                theta[i] = theta[i] + lasso
            else:
                theta[i] = 0
        
        #training error
        diff = np.dot(x,theta) - y
        training_error[iteration_idx] = np.sqrt((np.inner(diff,diff) + np.sum(np.abs(theta)))/ sample_dim)
        diff = np.dot(x_test,theta) - y_test
        testing_error[iteration_idx] = np.sqrt((np.inner(diff,diff) + np.sum(np.abs(theta)))/ sample_dim)
        
        #check whether it's time to stop
        #if np.linalg.norm(old_theta-theta,1) < precision:
        #    print "break"
        #    break
        
        #count nonzero theta
        #if num_exp_idx == num_exp - 1:
        #    nonzero_theta[iteration_idx] = np.count_nonzero(theta)
        
        iteration_idx += 1
        
    #try to find important features
    important_feature = np.argsort(np.abs(theta))[-num_largest_feature:]
    for i in range(0,num_largest_feature):
        feature_selected_count[important_feature[i]] += 1

stats = ['Constant','G','AB','R','H','2B','3B','HR','RBI','BB','SO','SB','CS','AVG','OBP','SLG','OPS']
important_feature = np.argsort(feature_selected_count)[-num_largest_feature:]
print feature_selected_count
for i in range(len(important_feature)):
    print stats[important_feature[i]]
