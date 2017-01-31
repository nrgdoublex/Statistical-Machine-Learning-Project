import numpy as np
import matrix_operation as mo


#generate matrix
input = open("training_data.txt")
data_matrix = []
lines = input.readlines()
for line in lines:
    list = line.rstrip('\n').split(',')
    list = map(float,list)
    data_matrix.append(list)
    
feature_dim = 16
    
x = np.array(mo.col_submatrix(data_matrix,0,feature_dim))
y = np.array(mo.getcolumnvector(data_matrix,feature_dim))
x_min = np.min(x, axis=0)
x_max = np.max(x, axis=0)
x = np.concatenate((np.ones((x.shape[0],1)),(x - x_min) / (x_max - x_min)),axis=1)
y_min = np.min(y, axis=0)
y_max = np.max(y, axis=0)
y = (y - y_min) / (y_max - y_min)

theta = np.dot(np.linalg.pinv(x),np.transpose(y))
diff = np.subtract(y,np.sum(np.multiply(x,theta),axis=1))
#print np.inner(diff,diff)
#print diff
#print '\n' 

#testing
testset = open("testing_data.txt")
data_matrix = []
lines = testset.readlines()
for line in lines:
    list = line.rstrip('\n').split(',')
    list = map(float,list)
    data_matrix.append(list)
x_test = np.array(mo.col_submatrix(data_matrix,0,feature_dim))
y_test = np.array(mo.getcolumnvector(data_matrix,feature_dim))
x_test = np.concatenate((np.ones((x_test.shape[0],1)),(x_test - x_min) / (x_max - x_min)),axis=1)
y_test = (y_test - y_min) / (y_max - y_min)
diff = np.subtract(y_test,np.sum(np.multiply(x_test,theta),axis=1))
print np.sqrt( np.inner(diff,diff)/174)
#print diff
#print '\n' 