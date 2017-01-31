import numpy as np
import random
import math

#generate matrix
input = open("training_data.txt")
data_matrix = []
lines = input.readlines()
for line in lines:
    list = line.rstrip('\n').split(',')
    list = [1.0] + map(float,list)
    data_matrix.append(list)
    
#parameters
dim = 17
learning_rate = 0.1
precision = 0.01
iteration = 1  

#normalize data-both x and y
max_vec = np.zeros(dim+1)
min_vec = np.empty(dim+1)
min_vec.fill(np.inf)
for vec in data_matrix:
    for idx in range(1,dim+1):
        if vec[idx] > max_vec[idx]:
            max_vec[idx] = vec[idx]
        if vec[idx] < min_vec[idx]:
            min_vec[idx] = vec[idx]
for vec in data_matrix:
    for idx in range(1,dim+1):
        vec[idx] = (vec[idx] - min_vec[idx])/(max_vec[idx]-min_vec[idx])
        

#main stochastic gradient descent algorithm
theta = np.random.rand(dim)
gradient = np.zeros(dim)
while iteration < 10000:
    old_theta = theta
    if iteration != 1 and np.mod(np.log10(iteration),1) == 0:
        learning_rate *= 0.5
    #shuffle data
    random.shuffle(data_matrix)
    #get gradient and renew theta
    for vector in data_matrix:
        y = vector[dim]
        x = vector[0:dim]
        diff = y - np.dot(x,theta)
        for idx in range(0,dim):
            if idx != 0:
                gradient[idx] = (-2) * learning_rate * x[idx] * diff
            else:
                gradient[idx] = (-2) * learning_rate * diff
        theta = np.subtract(theta,gradient)
    
    #check whether it's time to stop
    if np.linalg.norm(old_theta-theta,1) < precision:
        print "break"
        break
    
    iteration += 1
    
for vector in data_matrix:
    y = vector[dim]
    x = vector[0:dim]
    diff = np.absolute(y - np.dot(x,theta))
    print diff

#close file
input.close()
