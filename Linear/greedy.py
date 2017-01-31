import numpy as np
import matplotlib.pyplot as plt
import matrix_operation as mo
import input_data as id
import pylab

#number of features    
feature_dim = 16

#training set
x,y,x_min,x_max,y_min,y_max = id.read_trainingdata_MS(feature_dim)
#testing set
x_test,y_test = id.read_testingdata_MS(feature_dim, x_min, x_max, y_min, y_max)

#stats
stats = ['G','AB','R','H','2B','3B','HR','RBI','BB','SO','SB','CS','AVG','OBP','SLG','OPS','Constant']
#max number of features to select
max_features = 17

#size of training set
training_size = x.shape[0]
testing_size = x_test.shape[0]


#store loss
training_error = np.zeros(max_features)
testing_error = np.zeros(max_features)

#greedy algorithm
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
    training_error[num_feature_selected] = np.sqrt(np.inner(training_diff,training_diff) / training_size)
    testing_diff = y_test - np.dot(subx_test,theta)
    testing_error[num_feature_selected] = np.sqrt(np.inner(testing_diff,testing_diff) / testing_size)
    
    #marked selected
    feature_selected[currentidx] = 1
    
    num_feature_selected += 1
# plot
plt.plot(range(0,max_features),training_error,label='training')
plt.plot(range(0,max_features),testing_error,label='testing')
plt.title('error vs. number of features in greedy method')
plt.xlabel('number of features chosen')
plt.ylabel('Root-Mean-Square Error')
plt.legend(loc="upper right")
plt.savefig("greedy.png")

#save data
with open("greedy.txt",'w') as f:
    for idx in range(0,len(training_error)):
        f.write('%f,' %training_error[idx])
    f.write('\n')
    for idx in range(0,len(testing_error)):
        f.write('%f,' %testing_error[idx])
    f.write('\n')