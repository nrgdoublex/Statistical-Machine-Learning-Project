import numpy as np
import matrix_operation

#generate matrix
#input = open("training_data.txt")
#data_matrix = []
#lines = input.readlines()
#for line in lines:
#    list = line.rstrip('\n').split(',')
#    list = [1.0] + map(float,list)
#    data_matrix.append(list)

#print matrix_operation.row_submatrix(data_matrix, 0, 3)

#matrix = [[1,2,3],[4,5,6]]
#matrix2 = [[3,2,1],[6,5,4]]
#print np.subtract(matrix2,matrix)
#print np.transpose(np.array(np.sum(matrix,axis=1),ndmin=2))
#matrix = np.dot(matrix,[2,3,4])
#print matrix

vec = [1,-2,4,3,-4]
print np.sign(vec)