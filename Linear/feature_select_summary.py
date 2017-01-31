import numpy as np
import matplotlib.pyplot as plt
import matrix_operation as mo
import pylab

train_greedy = []
test_greedy = []
train_forward = []
test_forward = []
train_myopic = []
test_myopic = []

with open('greedy.txt') as f:
    line = f.readline()
    line = line.rstrip('\n').rstrip(',').split(',')
    train_greedy = map(float,line)
    line = f.readline()
    line = line.rstrip('\n').rstrip(',').split(',')
    test_greedy = map(float,line)
with open('forward-fitting.txt') as f:
    line = f.readline()
    line = line.rstrip('\n').rstrip(',').split(',')
    train_forward = map(float,line)
    line = f.readline()
    line = line.rstrip('\n').rstrip(',').split(',')
    test_forward = map(float,line)
with open('myopic-fitting.txt') as f:
    line = f.readline()
    line = line.rstrip('\n').rstrip(',').split(',')
    train_myopic = map(float,line)
    line = f.readline()
    line = line.rstrip('\n').rstrip(',').split(',')
    test_myopic = map(float,line)
    
#Plot
lw=2
plt.plot(range(0,len(train_greedy)), train_greedy, lw=lw, color='green', label='greedy(training)')
plt.plot(range(0,len(train_forward)), train_forward, lw=lw, color='orange', label='forward(training)')
plt.plot(range(0,len(train_myopic)), train_myopic, lw=lw, color='yellow', label='myopic(training)')

plt.plot(range(0,len(test_greedy)), test_greedy, lw=lw, color='blue', label='greedy(test)')
plt.plot(range(0,len(test_forward)), test_forward, lw=lw, color='cyan', label='forward(test)')
plt.plot(range(0,len(test_myopic)), test_myopic, lw=lw, color='magenta', label='myopic(test)')
        
plt.ylim([0.1,0.25])
plt.ylabel('Root-Mean-Square Error')
plt.xlabel('Number of features')
plt.title('Error of three methods')
plt.legend(loc="upper right")
plt.savefig("feature_select_summary.png")