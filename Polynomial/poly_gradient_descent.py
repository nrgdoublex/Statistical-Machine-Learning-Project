import numpy as np
import random
import math
import matrix_operation as mo
import matplotlib.pyplot as plt
import matplotlib.axes as axis
import input_data as id
import CV as cv

#parameters
learning_rate_set = [0.00005,0.00002,0.00001] #around 0.0001 level
precision = 0.0001
max_iteration = 1000

#we run for each degree
for deg in range(2,4):
    if deg == 1:
        training_file = "training_data.txt"
        testing_file = "testing_data.txt"
    elif deg == 2:
        training_file = "training_data_poly2.txt"
        testing_file = "testing_data_poly2.txt"
    elif deg == 3:
        training_file = "training_data_poly3.txt"
        testing_file = "testing_data_poly3.txt"
        
    
    #training set for linear regression
    x,y,x_min,x_max,y_min,y_max = id.read_trainingdata(training_file)
    #testing set for linear regression
    x_test,y_test = id.read_testingdata(testing_file, x_min, x_max, y_min, y_max)
    x = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
    x_test = np.concatenate((np.ones((x_test.shape[0],1)),x_test),axis=1)

    training_error = np.zeros((len(learning_rate_set),max_iteration))
    testing_error = np.zeros((len(learning_rate_set),max_iteration))
    learning_rate_idx = 0

    #train with different learning rate
    for learning_rate in learning_rate_set:
        #main gradient descent algorithm
        sample_dim = x.shape[0]
        theta = np.random.rand(x.shape[1]) #constant feature
        gradient = np.zeros(x.shape[1])
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
                print("break")
                break
            
            iteration_idx += 1
            
        learning_rate_idx += 1
            
    #plot
    plt.figure()
    for i in range(0,len(learning_rate_set)):
        plt.plot(np.arange(0,max_iteration),training_error[i],label='LR=%0.5f' %learning_rate_set[i])
    plt.axes().set_xscale('log')
    plt.title('Poly. regre. of Degree %d with GD for training' %deg)
    plt.xlabel('Number of iterations')
    plt.ylabel('Root-Mean-Square Error')
    plt.legend(loc="upper right")
    plt.savefig("Poly%dgradient_descent_training.png" %deg)
    
    plt.figure()
    for i in range(0,len(learning_rate_set)):
        plt.plot(np.arange(0,max_iteration),testing_error[i],label='LR=%0.5f' %learning_rate_set[i])
    plt.axes().set_xscale('log')
    plt.title('Poly. regre. of degree %d with GD for testing' %deg)
    plt.xlabel('Number of iterations')
    plt.ylabel('Root-Mean-Square Error')
    plt.legend(loc="upper right")
    plt.savefig("Poly%dgradient_descent_test.png" %deg)
