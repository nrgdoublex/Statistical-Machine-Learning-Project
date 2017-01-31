from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matrix_operation as mo
import input_data as id
import pylab
from scipy.stats import t

#number of features    
feature_dim = 16 
#max number of features to select
max_features = feature_dim + 1
#max_features = 

def greedy_selection(x,y,x_test,y_test,max_features):
    #store loss
    training_error = 0
    testing_error = 0
    #data matrix
    subx = []
    subx_test = []
    #number of current features
    num_feature_selected = 0
    #array to store whether a feature is selected
    feature_selected = np.zeros(feature_dim+1)
    while num_feature_selected < max_features:
        currentloss = np.Inf
        currentidx = 0
        currenttheta = []
        #choose features that are still unselected
        for idx in range(0,feature_dim+1):
            if feature_selected[idx] == 0:
                if idx == feature_dim:
                    col = np.ones(x.shape[0])
                else:
                    col = x[:,idx]
                #if empty features
                if num_feature_selected == 0:
                    tempx = np.transpose(np.array(col,ndmin=2))
                else:
                    tempx = np.concatenate((subx,np.transpose(np.array(col,ndmin=2))),axis=1)
                temptheta = np.dot(np.linalg.pinv(tempx),y)
                tempdiff = y - np.dot(tempx,temptheta)
                temploss = np.inner(tempdiff,tempdiff)
                if temploss < currentloss:
                    currentloss = temploss
                    currentidx = idx
                    currenttheta = temptheta
        
        #add new feature
        if currentidx == feature_dim:
            new_feature_train = np.ones(x.shape[0])
            new_feature_test = np.ones(x_test.shape[0])
        else:
            new_feature_train = x[:,currentidx]
            new_feature_test = x_test[:,currentidx]
        #if empty features
        if num_feature_selected == 0:
            subx = np.transpose(np.array(new_feature_train,ndmin=2))
            subx_test = np.transpose(np.array(new_feature_test,ndmin=2))
        else:
            subx = np.concatenate((subx,np.transpose(np.array(new_feature_train,ndmin=2))),axis=1)
            subx_test = np.concatenate((subx_test,np.transpose(np.array(new_feature_test,ndmin=2))),axis=1)
                    
        #calculate error
        theta = np.dot(np.linalg.pinv(subx),y)
        training_diff = y - np.dot(subx,theta)
        training_error = np.sqrt(np.inner(training_diff,training_diff) / training_size)
        testing_diff = y_test - np.dot(subx_test,theta)
        testing_error = np.sqrt(np.inner(testing_diff,testing_diff) / testing_size)
        
        #marked selected
        feature_selected[currentidx] = 1
        
        num_feature_selected += 1
    return testing_error

def forward_selection(x,y,x_test,y_test,max_features):
    #store loss
    training_error = 0
    testing_error = 0
    #data matrix
    subx = []
    subx_test = []
    #number of current features
    num_feature_selected = 0
    #array to store whether a feature is selected
    feature_selected = np.zeros(feature_dim+1)
    #initialize theta
    theta = []
    while num_feature_selected < max_features:
        currenttheta = 0
        currentidx = 0
        currentJ = np.Inf
        
        #subx * theta
        if num_feature_selected == 0:
            previous_prediction = np.zeros(x.shape[0])
        else:
            previous_prediction = np.sum(np.multiply(subx,np.transpose(np.array(theta,ndmin=2))),axis=0)
        previous_diff = np.subtract(y,previous_prediction)
        #choose features that are still unselected
        for idx in range(0,feature_dim+1):
            if feature_selected[idx] == 0:
                #the feature column
                if idx == feature_dim:
                    col = np.ones(x.shape[0])
                else:
                    col = x[:,idx]
                #print col
                # new theta
                temptheta = np.inner(previous_diff,col) / np.inner(col,col)
                #print temptheta
                current_prediction = np.subtract(previous_diff, col*temptheta)
                tempJ = 0.5 * np.inner(current_prediction,current_prediction)
    
                if tempJ < currentJ:
                    currentJ = tempJ
                    currentidx = idx
                    currenttheta = temptheta
                    
        #add new feature
        if currentidx == feature_dim:
            new_feature_train = np.array(np.ones(x.shape[0]),ndmin=2)
            new_feature_test = np.array(np.ones(x_test.shape[0]),ndmin=2)
        else:
            new_feature_train = np.array(x[:,currentidx],ndmin=2)
            new_feature_test = np.array(x_test[:,currentidx],ndmin=2)
        if subx == []:
            subx = new_feature_train
            subx_test = new_feature_test
        else:
            subx = np.concatenate((subx,new_feature_train),axis=0)
            subx_test = np.concatenate((subx_test,new_feature_test),axis=0)
        #renew theta
        theta.append(currenttheta)
        
        #print error
        train_predict = np.sum(np.multiply(subx,np.transpose(np.array(theta,ndmin=2))),axis=0)
        train_diff = np.subtract(y,train_predict)
        training_error = np.sqrt(np.inner(train_diff,train_diff) / training_size)
        test_predict = np.sum(np.multiply(subx_test,np.transpose(np.array(theta,ndmin=2))),axis=0)
        test_diff = np.subtract(y_test,test_predict)
        testing_error = np.sqrt(np.inner(test_diff,test_diff) / testing_size)
        
        feature_selected[currentidx] = 1
        
        num_feature_selected += 1
        
    return testing_error

def myopic_selection(x,y,x_test,y_test,max_features):
    #store loss
    training_error = 0
    testing_error = 0
    #data matrix
    subx = []
    subx_test = []
    #number of current features
    num_feature_selected = 0
    #array to store whether a feature is selected
    feature_selected = np.zeros(feature_dim+1)
    #initialize theta
    theta = []
    
    while num_feature_selected < max_features:
        currenttheta = 0
        currentidx = 0
        currentDJ = 0
        
        #subx * theta
        if num_feature_selected == 0:
            previous_prediction = np.zeros(x.shape[0])
        else:
            previous_prediction = np.sum(np.multiply(subx,np.transpose(np.array(theta,ndmin=2))),axis=0)
        previous_diff = np.subtract(y,previous_prediction)
        #choose features that are still unselected
        for idx in range(0,feature_dim+1):
            if feature_selected[idx] == 0:
                #the feature column
                if idx == feature_dim:
                    col = np.ones(x.shape[0])
                else:
                    col = x[:,idx]
                #print col
                # get DJ
                tempDJ = np.abs(np.inner(previous_diff,col))
    
                if tempDJ > currentDJ:
                    currentDJ = tempDJ
                    currentidx = idx
                    
        ##add new feature
        if currentidx == feature_dim:
            new_feature_train = np.ones(x.shape[0])
            new_feature_test = np.ones(x_test.shape[0])
        else:
            new_feature_train = x[:,currentidx]
            new_feature_test = x_test[:,currentidx]
        if subx == []:
            subx = np.array(new_feature_train,ndmin=2)
            subx_test = np.array(new_feature_test,ndmin=2)
        else:
            subx = np.concatenate((subx,np.array(new_feature_train,ndmin=2)),axis=0)
            subx_test = np.concatenate((subx_test,np.array(new_feature_test,ndmin=2)),axis=0)
        #renew theta
        currenttheta = np.inner(previous_diff,new_feature_train) / np.inner(new_feature_train,new_feature_train)
        theta.append(currenttheta)
        
        #print error
        train_predict = np.sum(np.multiply(subx,np.transpose(np.array(theta,ndmin=2))),axis=0)
        train_diff = np.subtract(y,train_predict)
        training_error = np.sqrt(np.inner(train_diff,train_diff) / training_size)
        test_predict = np.sum(np.multiply(subx_test,np.transpose(np.array(theta,ndmin=2))),axis=0)
        test_diff = np.subtract(y_test,test_predict)
        testing_error = np.sqrt(np.inner(test_diff,test_diff) / testing_size)
        
        feature_selected[currentidx] = 1
        
        num_feature_selected += 1
        
    return testing_error

#cross validation indices
def KFold(df,k):
    quan = np.zeros(k+1)
    total_size = len(df)
    subset_size = np.floor(total_size / k)
    residue = np.remainder(total_size,k)
    end_idx = 0
    for i in range(1,k+1):
        if i <= residue:
            quan[i] = end_idx + subset_size + 1
            end_idx += (subset_size + 1)
        else:
            quan[i] = end_idx + subset_size
            end_idx += subset_size
    return [int(i) for i in quan]

def shuffle(df, array):   
    shuffled_array = np.zeros(len(df),dtype=np.int32)
    
    randomize = np.arange(len(df))
    np.random.shuffle(randomize)
    shuffled_df = df[randomize]
    for i in range(0,len(randomize)):
        shuffled_array[i] = array[randomize[i]]
    return shuffled_df,shuffled_array

def one_tailed_ttest(data1,data2):
    len_data1 = len(data1)
    len_data2 = len(data2)
    mean_data1 = np.average(data1)
    mean_data2 = np.average(data2)
    var_data1 = np.var(data1)
    var_data2 = np.var(data2)
    
    t_score = (mean_data1-mean_data2) / np.sqrt(var_data1/len_data1+var_data2/len_data2)
    df = np.square(var_data1/len_data1+var_data2/len_data2) / \
        (np.square(var_data1/len_data1)/(len_data1-1) + np.square(var_data2/len_data2)/(len_data2-1))
    p_value = t.sf(t_score, df)
    
    return t_score,p_value


#stats
stats = ['G','AB','R','H','2B','3B','HR','RBI','BB','SO','SB','CS','AVG','OBP','SLG','OPS','Constant']


#training set
x,y,x_min,x_max,y_min,y_max = id.read_trainingdata_MS(feature_dim)
#testing set
x_test,y_test = id.read_testingdata_MS(feature_dim, x_min, x_max, y_min, y_max)

#size of training set
training_size = x.shape[0]
testing_size = x_test.shape[0]

kfold = 20
algorithm_set = ['greedy','forward','myopic']

#storing validation error
greedy_error = np.zeros(kfold)
forward_error = np.zeros(kfold)
myopic_error = np.zeros(kfold)

for algorithm in algorithm_set:
    KFold_indices = KFold(x,kfold)
    
    #shuffle data
    x_shuffle,y_shuffle = shuffle(x,y)
    #divide data into training and validation set
    testing_error = 0
    for i in range(0,kfold):
        validation_data = x_shuffle[KFold_indices[i]:KFold_indices[i+1]]
        validation_target = y_shuffle[KFold_indices[i]:KFold_indices[i+1]]
        if i == 0:
            training_data = x_shuffle[KFold_indices[i+1]:]
            training_target = y_shuffle[KFold_indices[i+1]:]
        elif i == kfold - 1:
            training_data = x_shuffle[:KFold_indices[i]]
            training_target = y_shuffle[:KFold_indices[i]]
        else:
            training_data = np.append(x_shuffle[:KFold_indices[i]],x_shuffle[KFold_indices[i+1]:],axis=0)
            training_target = np.append(y_shuffle[:KFold_indices[i]],y_shuffle[KFold_indices[i+1]:],axis=0)
        
        #algorithm
        if algorithm == 'greedy':
            greedy_error[i] = greedy_selection(training_data,training_target, \
                                validation_data,validation_target,max_features)
        elif algorithm == 'forward':
            forward_error[i] = forward_selection(training_data,training_target, \
                                validation_data,validation_target,max_features)
        elif algorithm == 'myopic':
            myopic_error[i] = myopic_selection(training_data,training_target, \
                                validation_data,validation_target,max_features)
    
    #test
    if algorithm == 'greedy':
        testing_error = greedy_selection(x_shuffle,y_shuffle, \
                                x_test,y_test,max_features)
    elif algorithm == 'forward':
        testing_error = forward_selection(x_shuffle,y_shuffle, \
                                x_test,y_test,max_features)
    elif algorithm == 'myopic':
        testing_error = myopic_selection(x_shuffle,y_shuffle, \
                                x_test,y_test,max_features)
    print(testing_error)
    
#t-test
print(one_tailed_ttest(greedy_error,forward_error))
print(one_tailed_ttest(greedy_error,myopic_error))
print(one_tailed_ttest(forward_error,myopic_error))
            
            
            




