import numpy as np
import random
import math
import matrix_operation as mo
import matplotlib.pyplot as plt
import matplotlib.axes as axis
import input_data as id

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

#parameters
feature_dim = 16

#training set
x,y,x_min,x_max,y_min,y_max = id.read_trainingdata_MS(feature_dim)
x = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)  
#testing set
x_test,y_test = id.read_testingdata_MS(feature_dim, x_min, x_max, y_min, y_max)
x_test = np.concatenate((np.ones((x_test.shape[0],1)),x_test),axis=1) 
    
#parameters
kfold_set = [2,5,10,20,50,100,200,x.shape[0]]

#store mean and sd of each kfold
training_mean = np.zeros(len(kfold_set))
training_sd = np.zeros(len(kfold_set))
validation_mean = np.zeros(len(kfold_set))
validation_sd = np.zeros(len(kfold_set))
test_mean = np.zeros(len(kfold_set))
test_sd = np.zeros(len(kfold_set))


#train with different learning rate
kfold_set_idx = 0
for num_kfold in kfold_set:
    print(num_kfold)
    KFold_indices = KFold(x,num_kfold)
    
    #shuffle data
    x_shuffle,y_shuffle = shuffle(x,y)
    
    #data structure to store training and validation error
    training_error = []
    validation_error = []
    test_error = []
    for i in range(0,num_kfold):
        validation_data = x_shuffle[KFold_indices[i]:KFold_indices[i+1]]
        validation_target = y_shuffle[KFold_indices[i]:KFold_indices[i+1]]
        if i == 0:
            training_data = x_shuffle[KFold_indices[i+1]:]
            training_target = y_shuffle[KFold_indices[i+1]:]
        elif i == num_kfold - 1:
            training_data = x_shuffle[:KFold_indices[i]]
            training_target = y_shuffle[:KFold_indices[i]]
        else:
            training_data = np.append(x_shuffle[:KFold_indices[i]],x_shuffle[KFold_indices[i+1]:],axis=0)
            training_target = np.append(y_shuffle[:KFold_indices[i]],y_shuffle[KFold_indices[i+1]:],axis=0)
            
        #closed-form solution
        theta = np.dot(np.linalg.pinv(training_data),np.transpose(training_target))
            
        #training error
        diff = np.dot(training_data,theta) - training_target
        training_error.append(np.sqrt(np.inner(diff,diff) / training_data.shape[0]))
        diff = np.dot(validation_data,theta) - validation_target
        validation_error.append(np.sqrt(np.inner(diff,diff) / validation_data.shape[0]))
        diff = np.dot(x_test,theta) - y_test
        test_error.append(np.sqrt(np.inner(diff,diff) / x_test.shape[0]))

    #statistics
    training_mean[kfold_set_idx] = np.mean(training_error)
    training_sd[kfold_set_idx] = np.std(training_error)
    validation_mean[kfold_set_idx] = np.mean(validation_error)
    validation_sd[kfold_set_idx] = np.std(validation_error)
    test_mean[kfold_set_idx] = np.mean(test_error)
    test_sd[kfold_set_idx] = np.std(test_error)
    
    kfold_set_idx += 1
        
#plot
plt.figure()
plt.plot(kfold_set,training_mean,label='training')
plt.plot(kfold_set,validation_mean,label='validation')
plt.plot(kfold_set,test_mean,label='test')
plt.axes().set_xscale('log')
plt.title('Mean of error with different k-fold cross validation')
plt.xlabel('Number of k folds')
plt.ylabel('Root-Mean-Square Error')
plt.legend(loc="upper right")
plt.savefig("compare_cross_validation_mean.png")

plt.figure()
plt.plot(kfold_set,training_sd,label='training')
plt.plot(kfold_set,validation_sd,label='validation')
plt.plot(kfold_set,test_sd,label='test')
plt.axes().set_xscale('log')
plt.title('Standard deviation of error with different k-fold cross validation')
plt.xlabel('Number of k folds')
plt.ylabel('Root-Mean-Square Error')
plt.legend(loc="upper left")
plt.savefig("compare_cross_validation_sd.png")

