import pandas as pd
import numpy as np
import statistics
import tree
import timeit
from scipy.optimize import fmin

M = 100
MAX_LEAVES = 20
TREE_DEPTH = 2
LEARNING_RATE = 0.1

def loss_function(y, x):
    return (0.5 * (y - x) ** 2)

def find_prediction(observed):
    return statistics.mean(observed)

def load_data(filename):
    data = pd.read_csv(filename, delim_whitespace=' ', header=None)
    return data.values.tolist()

data = load_data('housing.csv')
length = round(0.75 * len(data))

x_train = data[:length]
x_test = data[length:]

y_train = list()
y_test = list()
# remove y from x data
[y_train.append(j.pop(len(data[0])-1)) for j in x_train]
[y_test.append(j.pop(len(data[0])-1)) for j in x_test]

trees = list(); # list for all generated trees

f_0 = statistics.mean(y_train) # first predicted value
f_m = [f_0] * len(y_train) # predicted values

for m in range(1, M): # Iteration loop
    start = timeit.default_timer() # timestamp

    r = list() # current residuals
    for i in range(0, len(y_train)): # Residuals for every data row loop
        r.append(y_train[i] - f_m[i]) # calculate current residuals


    decision_tree = tree.build_tree(x_train, r, TREE_DEPTH, MAX_LEAVES) # create current decision tree
    trees.append(decision_tree) # save decision tree

    for i in range(0, len(f_m)): # update predicted values
        r_value = tree.find_value(decision_tree, x_train[i]) # find residul output value from tree
        f_m[i] += (LEARNING_RATE * r_value) # add residual value with learning rate to estimation function

    # timestamp
    end = timeit.default_timer()
    print('Iteration: ' + str(m) + ', time: ' + str(end - start))

test_predicted = [f_0] * len(y_test) # first value of predicted

for m_tree in trees: # add predicted values from every tree
    for i in range(0, len(test_predicted)):
        test_predicted[i] += LEARNING_RATE * tree.find_value(m_tree, x_test[i]) # add residual value from tree with learning rate


RMSE = round((np.square(np.subtract(y_test,test_predicted)).mean()) ** (0.5), 2)
print('RMSE: ' + str(RMSE))