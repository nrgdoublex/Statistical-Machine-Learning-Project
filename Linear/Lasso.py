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
lasso_lambda = [0,0.001,0.01,0.1,1,10,100,1000]

#count nonzero theta
nonzero_theta = np.zeros(max_iteration)
        
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

#main stochastic gradient descent algorithm
sample_dim = x.shape[0]
for lasso in lasso_lambda:
    theta = np.random.rand(feature_dim+1) #constant feature
    gradient = np.zeros(feature_dim+1)
    iteration_idx = 0
    training_error = np.zeros(max_iteration)
    testing_error = np.zeros(max_iteration)
    while iteration_idx < max_iteration:
        old_theta = theta
        #get gradient and renew theta
        diff = np.dot(x,theta) - y
        #calculate gradient
        gradient = (2) * np.dot(np.transpose(x),diff)
        theta = theta - learning_rate * (gradient)
        #soft-thresholding
        for i in range(0,len(theta)):
            if theta[i] > lasso:
                theta[i] = theta[i] - lasso
            elif theta[i] < (-1) * lasso:
                theta[i] = theta[i] + lasso
            else:
                theta[i] = 0
                
        #count nonzero theta
        nonzero_theta[iteration_idx] = np.count_nonzero(theta)
        
        #training error
        diff = np.dot(x,theta) - y
        training_error[iteration_idx] = np.sqrt((0.5*np.inner(diff,diff) + np.sum(np.abs(theta)))/ sample_dim)
        diff = np.dot(x_test,theta) - y_test
        testing_error[iteration_idx] = np.sqrt((0.5*np.inner(diff,diff) + np.sum(np.abs(theta)))/ sample_dim)
        
        #check whether it's time to stop
        #if np.linalg.norm(old_theta-theta,1) < precision:
        #    print "break"
        #    break
        
        iteration_idx += 1
    
    #try to find important features
    #print theta
    #rint np.argsort(theta)[-5:]

# plot
    lb = '$\lambda$=' + str(lasso)
    ax1.plot(np.arange(0,max_iteration),testing_error,label=lb)
    ax2.plot(np.arange(0,max_iteration),nonzero_theta,label=lb)
    
    
ax1.set_xscale('log')
fig1.suptitle('Test error vs Iterations')
ax1.set_xlabel('Number of iterations')
ax1.set_ylabel('Root Mean Square Error')
ax1.legend(loc="upper left", bbox_to_anchor=[0, 1],
           ncol=2, shadow=True, title="Test Error for each $\lambda$", fancybox=True)
fig1.savefig("Lasso.png")

ax2.set_xscale('log')
fig2.suptitle('Number of nonzero features vs Iterations')
ax2.set_xlabel('Number of iterations')
ax2.set_ylabel('Number of nonzero features')
ax2.set_ylim([-1,30])
ax2.legend(loc="upper left", bbox_to_anchor=[0, 1],
           ncol=2, shadow=True, title="Number of nonzero features for each $\lambda$", fancybox=True)
fig2.savefig("Lasso_Feature.png")
