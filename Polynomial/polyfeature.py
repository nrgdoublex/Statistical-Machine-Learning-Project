import numpy as np
import matrix_operation as mo

feature_dim = 16

def poly2_feature(input_file,output_file):
    file = open(input_file)
    output = open(output_file,'w')
    matrix = []
    lines = file.readlines()
    for line in lines:
        data_list = line.rstrip('\n').split(',')
        data_list = map(float,data_list)
        y = data_list.pop()
        num_features = len(data_list)
        output_list = list(data_list)
        for i in range(0,num_features):
            for j in range(i,num_features):
                output_list.append(data_list[i]*data_list[j])
         
        for idx in range(0,len(output_list)):
                output.write("%s," %output_list[idx])
        output.write("%s\n" %y)
        
def poly3_feature(input_file,output_file):
    file = open(input_file)
    output = open(output_file,'w')
    matrix = []
    lines = file.readlines()
    for line in lines:
        data_list = line.rstrip('\n').split(',')
        data_list = map(float,data_list)
        y = data_list.pop()
        num_features = len(data_list)
        output_list = list(data_list)
        
        #2-order features
        for i in range(0,num_features):
            for j in range(i,num_features):
                output_list.append(data_list[i]*data_list[j])
        #3-order features  
        for i in range(0,num_features):
            for j in range(i,num_features):
                for k in range(j,num_features):
                    output_list.append(data_list[i]*data_list[j]*data_list[k])
         
        for idx in range(0,len(output_list)):
                output.write("%s," %output_list[idx])
        output.write("%s\n" %y)      
        
poly2_feature("training_data.txt","training_data_poly2.txt")
poly2_feature("testing_data.txt","testing_data_poly2.txt")
poly3_feature("training_data.txt","training_data_poly3.txt")
poly3_feature("testing_data.txt","testing_data_poly3.txt")

    
