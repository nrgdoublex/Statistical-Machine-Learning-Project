import numpy as np
import matplotlib.pyplot as plt
import matrix_operation as mo
import input_data as id

#number of features    
feature_dim = 16 #including constant feature

#training set
x,y,x_min,x_max,y_min,y_max = id.read_trainingdata_MS(feature_dim)
#testing set
x_test,y_test = id.read_testingdata_MS(feature_dim, x_min, x_max, y_min, y_max)
#data set size
training_size = x.shape[0]
testing_size = x_test.shape[0]

#stats
stats = ['G','AB','R','H','2B','3B','HR','RBI','BB','SO','SB','CS','AVG','OBP','SLG','OPS','Constant']
#array to store whether a feature is selected
feature_selected = np.zeros(feature_dim+1)
#max number of features to select
max_features = 17
#number of current features
num_feature_selected = 0

#store loss
training_error = np.zeros(max_features)
testing_error = np.zeros(max_features)

selected_feature_order = np.zeros(feature_dim + 1)

#forward-fitting algorithm
subx = []
subx_test = []
#initialize theta
theta = []
#theta.append(np.inner(y,subx)[0]/np.inner(subx,subx)[0][0])
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
    training_error[num_feature_selected] = np.sqrt(np.inner(train_diff,train_diff) / training_size)
    test_predict = np.sum(np.multiply(subx_test,np.transpose(np.array(theta,ndmin=2))),axis=0)
    test_diff = np.subtract(y_test,test_predict)
    testing_error[num_feature_selected] = np.sqrt(np.inner(test_diff,test_diff) / testing_size)
    
    feature_selected[currentidx] = 1
    
    selected_feature_order[currentidx] = num_feature_selected + 1
    
    num_feature_selected += 1
    
print selected_feature_order
# plot
plt.plot(range(0,max_features),training_error,label='training')
plt.plot(range(0,max_features),testing_error,label='testing')
plt.title('error vs. number of features in myopic-fitting method')
plt.xlabel('number of features chosen')
plt.ylabel('Root-Mean-Square Error')
plt.legend(loc="upper right")
plt.savefig("myopic_fitting.png")

#save data
with open("myopic-fitting.txt",'w') as f:
    for idx in range(0,len(training_error)):
        f.write('%f,' %training_error[idx])
    f.write('\n')
    for idx in range(0,len(testing_error)):
        f.write('%f,' %testing_error[idx])
    f.write('\n')