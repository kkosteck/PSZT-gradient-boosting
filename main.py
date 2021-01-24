import pandas as pd
import numpy as np
import statistics
import tree
from scipy.optimize import fmin

ROWS = 506
M = 100
Y_ID = 13
TREE_DEPTH = 3

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
print(x_train[0][1])
# remove y from x data
[y_train.append(j.pop(Y_ID)) for j in x_train]
[y_test.append(j.pop(Y_ID)) for j in x_test]

f_m = statistics.mean(y_train)
residuals = list();

for m in range(1, M):
    r = list()
    for i in range(0, len(y_train)):
        r.append(y_train[i] - f_m)
        decision_tree = tree.build_tree(x_train, y_train, TREE_DEPTH)
    