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
learning_rate_set = [0.0002,0.0001,0.00005,0.00002,0.00001] #around 0.0001 level
precision = 0.0001
max_iteration = 1000

training_error = np.zeros((len(learning_rate_set),max_iteration))
testing_error = np.zeros((len(learning_rate_set),max_iteration))
learning_rate_idx = 0

plt.figure()
#train with different learning rate
for learning_rate in learning_rate_set:
    #main gradient descent algorithm
    sample_dim = x.shape[0]
    theta = np.random.rand(feature_dim+1) #constant feature
    gradient = np.zeros(feature_dim+1)
    iteration_idx = 0
    while iteration_idx < max_iteration:
        old_theta = theta
    #    if iteration_idx != 1 and np.mod(np.log10(iteration),1) == 0:
    #        learning_rate *= 0.5
        #get gradient and renew theta
        diff = np.dot(x,theta) - y
        gradient = (2) * np.dot(np.transpose(x),diff)
        #print gradient
        theta = theta - learning_rate * gradient
        
        #training error
        diff = np.dot(x,theta) - y
        training_error[learning_rate_idx][iteration_idx] = np.sqrt(np.inner(diff,diff) / sample_dim)
        diff = np.dot(x_test,theta) - y_test
        testing_error[learning_rate_idx][iteration_idx] = np.sqrt(np.inner(diff,diff) / sample_dim)
    
        
        #check whether it's time to stop
        if np.linalg.norm(old_theta-theta,1) < precision:
            print "break"
            break
        
        iteration_idx += 1
        
    learning_rate_idx += 1
        
#plot
for i in range(0,len(learning_rate_set)):
    plt.plot(np.arange(0,max_iteration),training_error[i],label='LR=%0.5f' %learning_rate_set[i])
plt.axes().set_xscale('log')
plt.title('Linear regression with gradient descent for training')
plt.xlabel('Number of iterations')
plt.ylabel('Root-Mean-Square Error')
plt.legend(loc="upper right")
plt.savefig("gradient_descent_training.png")

plt.figure()
for i in range(0,len(learning_rate_set)):
    plt.plot(np.arange(0,max_iteration),testing_error[i],label='LR=%0.5f' %learning_rate_set[i])
plt.axes().set_xscale('log')
plt.title('Linear regression with gradient descent for testing')
plt.xlabel('Number of iterations')
plt.ylabel('Root-Mean-Square Error')
plt.legend(loc="upper right")
plt.savefig("gradient_descent_test.png")
