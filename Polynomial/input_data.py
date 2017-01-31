import numpy as np
import matrix_operation as mo

test = False

def read_trainingdata(training_file):
    training = open(training_file)
    training_matrix = []
    lines = training.readlines()
    for line in lines:
        list = line.rstrip('\n').split(',')
        list = map(float,list)
        training_matrix.append(list)
    feature_dim = len(training_matrix[0]) - 1
    x = np.array(mo.col_submatrix(training_matrix,0,feature_dim))
    y = np.array(mo.getcolumnvector(training_matrix,feature_dim))
    if test == False:
        x_min = np.min(x, axis=0)
        x_max = np.max(x, axis=0)
        x = (x - x_min) / (x_max - x_min)
        y_min = np.min(y)
        y_max = np.max(y)
        y = (y - y_min) / (y_max - y_min)
    else:
        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0
    
    return x,y,x_min,x_max,y_min,y_max

def read_testingdata(testing_file,x_min,x_max,y_min,y_max):
    testing = open(testing_file)
    testing_matrix = []
    lines = testing.readlines()
    for line in lines:
        list = line.rstrip('\n').split(',')
        list = map(float,list)
        testing_matrix.append(list)
        feature_dim = len(testing_matrix[0]) - 1
    x_test = np.array(mo.col_submatrix(testing_matrix,0,feature_dim))
    y_test = np.array(mo.getcolumnvector(testing_matrix,feature_dim))
    if test == False:
        x_test = (x_test - x_min) / (x_max - x_min)
        y_test = (y_test - y_min) / (y_max - y_min)
    
    return x_test,y_test