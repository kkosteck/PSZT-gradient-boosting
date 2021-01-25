import pandas as pd
import numpy as np
import statistics
import tree
from scipy.optimize import fmin

M = 50
TREE_DEPTH = 3
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
    print(m)
    r = list()
    for i in range(0, len(y_train)): # Residuals for every data row loop
        r.append(y_train[i] - f_m[i])
    
    decision_tree = tree.build_tree(x_train, r, TREE_DEPTH) # create current decision tree
    trees.append(decision_tree) # save decision tree
    for i in range(0, len(f_m)): # update predicted values
        f_m[i] += LEARNING_RATE * tree.find_value(decision_tree, x_train[i])

test_predicted = [f_0] * len(y_test) # first value of predicted
errors = [0] * len(y_test)

for m_tree in trees: # add predicted values from every tree
    for i in range(0, len(test_predicted)):
        test_predicted[i] += LEARNING_RATE * tree.find_value(m_tree, x_train[i])
        errors[i] = abs(y_test[i] - test_predicted[i])

round_errors = [round(num, 2) for num in errors]
print(round_errors)