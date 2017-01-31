import numpy as np
from scipy import stats

#input matrix should be list of list, end_col is (last index)+1
def col_submatrix(matrix, start_col,end_col):
    submatrix = []
    for list in matrix:
        sublist = list[start_col:end_col]
        submatrix.append(sublist)
    return submatrix

#input matrix should be list of list, end_col is (last index)+1
def row_submatrix(matrix, start_col,end_col):
    submatrix = []
    for i in range(start_col,end_col):
        submatrix.append(matrix[i])
    return submatrix

def getcolumnvector(matrix,index):
    vec = np.zeros(len(matrix))
    for i in range(0,len(matrix)):
        vec[i] = matrix[i][index]
    return vec

def transpose(matrix):
    col_dim = len(matrix)
    row_dim = len(matrix[0])
    output = np.zeros((row_dim,col_dim))
    for i in range(0,row_dim):
        for j in range(0,col_dim):
            output[i][j] = matrix[j][i]
    return output

def multiply(matrix1,matrix2):
    row_dim = len(matrix1)
    col_dim = len(matrix2[0])
    matrix = np.zeros((row_dim,col_dim))
    for i in range(0,row_dim):
        for j in range(0,col_dim):
            row = matrix1[i]
            col = getcolumnvector(matrix2,j)
            matrix[i][j] = np.dot(row,col)
    return matrix

def pseudoinverse(matrix):
    return multiply(matrix, np.linalg.inv(multiply(matrix, transpose(matrix))))


def normalization(matrix):
    max_vec = np.amax(matrix, axis=0)
    min_vec = np.amin(matrix, axis=0)
    
#    mean = np.mean(matrix, axis=0)
#    stderr = stats.sem(matrix, axis=0)   
    output = (matrix - min_vec) / (max_vec - min_vec)
    return (output,max,min)
    