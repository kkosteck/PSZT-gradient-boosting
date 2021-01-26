import pandas as pd
import numpy as np
import statistics
import tree
import timeit
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error as MSE 
from sklearn import datasets 
import matplotlib.pyplot as plt

M = 400
MAX_LEAVES = 20
TREE_DEPTH = 4
LEARNING_RATE = 0.2
K_FOLD = 4
FILENAME = 'housing.csv'

def loss_function(y, x):
    return (0.5 * (y - x) ** 2)

def find_prediction(observed):
    return statistics.mean(observed)

def load_data():
    housing = datasets.load_boston() 
    x, y = housing.data, housing.target 
    x = np.array(x).tolist()
    y = np.array(y).tolist()

    return x, y

def split_data(x, y, k):
    test_start = round(len(y) / K_FOLD) * k
    test_end = round(len(y) / K_FOLD) * (k + 1)

    x_test, y_test = [],[]
    x_train = x.copy()
    y_train = y.copy()
    if k + 1 != K_FOLD:
        for i in range(test_start, test_end - 1):
            x_test.append(x_train.pop(i))
            y_test.append(y_train.pop(i))
    else:
        for i in range(test_start, test_end):
            x_test.append(x_train.pop())
            y_test.append(y_train.pop())

    return x_train, y_train, x_test, y_test

def sklearn_gradient_boosting(x_train, y_train, x_test, y_test):
    gbr = GradientBoostingRegressor(n_estimators = M, max_depth = TREE_DEPTH, learning_rate=LEARNING_RATE) 
    gbr.fit(x_train, y_train) 
    pred_y = gbr.predict(x_test) 

    np.array(pred_y).tolist()
    return pred_y

def gradient_boosting(x_train, y_train, x_test, y_test):

    trees = list() # list for all generated trees

    f_0 = statistics.mean(y_train) # first predicted value
    f_m = [f_0] * len(y_train) # predicted values

    for m in range(1, M): # Iteration loop

        # print('Iteration: ' + str(m))

        start = timeit.default_timer() # timestamp

        r = list() # current residuals
        for i in range(0, len(y_train)): # Residuals for every data row loop
            r.append(y_train[i] - f_m[i]) # calculate current residuals

        decision_tree = tree.build_tree(x_train, r, TREE_DEPTH, MAX_LEAVES) # create current decision tree
        trees.append(decision_tree) # save decision tree

        for i in range(0, len(f_m)): # update predicted values
            r_value = tree.find_value(decision_tree, x_train[i]) # find residul output value from tree
            f_m[i] += (LEARNING_RATE * r_value) # add residual value with learning rate to estimation function

    test_predicted = [f_0] * len(y_test) # first value of predicted

    for m_tree in trees: # add predicted values from every tree
        for i in range(0, len(test_predicted)):
            value = tree.find_value(m_tree, x_test[i])
            # print(x_test[i])
            # print(value)
            test_predicted[i] += LEARNING_RATE * value # add residual value from tree with learning rate

    return test_predicted

def calculate_parameters(observed, predicted):
    rmse = round(MSE(observed, predicted) ** (1 / 2), 3)
    minimum = round(min(predicted), 3)
    maximum = round(max(predicted), 3)
    median = round(statistics.median(predicted), 3)
    mean = round(statistics.mean(predicted), 3)

    return [rmse, minimum, maximum, median, mean]

def main():
    x, y = load_data()

    start = timeit.default_timer() # timestamp
    predictions = []
    y_tests = []
    for k in range(0, K_FOLD):
        print('GBR - iteration: ' + str(k+1))
        x_train, y_train, x_test, y_test = split_data(x, y, k)
        predictions.append(gradient_boosting(x_train, y_train, x_test, y_test))
        y_tests.append(y_test)
    end = timeit.default_timer() # timestamp
    execution_time = round(end - start, 3)

    start = timeit.default_timer() # timestamp
    sk_predictions = []
    for k in range(0, K_FOLD):
        print('sklearn - iteration: ' + str(k+1))
        x_train, y_train, x_test, y_test = split_data(x, y, k)
        sk_predictions.append(sklearn_gradient_boosting(x_train, y_train, x_test, y_test))
    end = timeit.default_timer() # timestamp
    sk_execution_time = round(end - start, 3)

    # average
    prediction = [0] * len(predictions[0])
    sk_prediction = [0] * len(sk_predictions[0])
    y_test = [0] * len(y_tests[0])
    for i in range(0, len(predictions[0])):
        for j in range(0, len(predictions)):
            prediction[i] += predictions[j][i]
            sk_prediction[i] += sk_predictions[j][i]
            y_test[i] += y_tests[j][i]
        prediction[i] /= 5
        sk_prediction[i] /= 5
        y_test[i] /= 5

    #   PLOTS
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    axes[0].set_title('Observed and predicted values')
    axes[0].plot(prediction, c='red')
    axes[0].plot(sk_prediction, c='green')
    axes[0].scatter(range(0, len(y_test)), y_test)
    axes[0].legend(('prediction','sklearn prediction','observed'))
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('value')

    errors, sk_errors = [], []
    for i in range(0, len(prediction)):
        errors.append(abs(prediction[i] - y_test[i]))
        sk_errors.append(abs(sk_prediction[i] - y_test[i]))
    
    plt.style.use('seaborn-deep')
    bins = np.linspace(0, 10, 30)
    axes[1].hist([errors, sk_errors], bins, label=['x', 'y'])
    axes[1].set_title('Histogram of residuals of observed and predicted values')
    axes[1].legend(('implementation','sklearn'))
    axes[1].set_xlabel('value')
    axes[1].set_ylabel('quantity')
    
    axes[0].set_position((.07, .20, .4, .65))
    axes[1].set_position((.55, .20, .4, .65))
    text = "Learning rate: " + str(LEARNING_RATE) + ", k-fold: " + str(K_FOLD) + ", iterations: " + str(M) + ", tree depth: " + str(TREE_DEPTH)
    plt.figtext(.5, .05, text , horizontalalignment ="center")
    plt.savefig('plot.png')

    values = calculate_parameters(y_test, prediction)
    values.append(execution_time)
    sk_values = calculate_parameters(y_test, sk_prediction)
    sk_values.append(sk_execution_time)
    df = pd.DataFrame({'projekt': values, 'sklearn': sk_values}) 
    df.to_excel('parameters.xlsx')



    print('My: ' + str(execution_time) + ', Sklearn: ' + str(sk_execution_time))

if __name__=="__main__": 
    main() 