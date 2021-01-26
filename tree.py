import statistics
import timeit

THRESHOLD_STEPS = 10

class Leaf:
    def __init__(self, y):
        self.value = statistics.mean(y)

class Decision_Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, x):
        val = x[self.column]

        return val <= self.value

def find_best_split(x, y):
    best_value = float("inf")
    best_question = Question(0, 0)
    n_features = len(x[0])

    for col in range(n_features):  # for each feature
        x_sorted, y_sorted = sort_column(x, y, col)
        current_value, threshold = best_threshold(x_sorted, y_sorted)

        if(threshold > x_sorted[-1]):
            continue

        # print('Column: ' + str(col) + ', SSR: ' + str(current_value) + ', Threshold: ' + str(threshold))

        question = Question(col, threshold)
        if current_value < best_value: 
            best_value = current_value
            best_question = question

    return best_question

def best_threshold(x, y):
    lowest_sum = float('inf')
    threshold = 0

    steps = calculate_steps(x)

    for i in steps:
        avg_before, avg_after = calculate_averages(x, y, i)
        sum_of_residuals = 0
        for j in range(0, len(y)):
            if x[j] < i:
                sum_of_residuals += ((y[j] - avg_before) ** 2)
            else:
                sum_of_residuals += ((y[j] - avg_after) ** 2)

        if sum_of_residuals < lowest_sum: 
            lowest_sum = sum_of_residuals
            threshold = i
    
    return lowest_sum, threshold

def calculate_steps(x):
    step = (x[-1] - x[0]) / THRESHOLD_STEPS + 1
    steps = []
    value = x[0]
    while value < x[-1]:
        value += step
        steps.append(value)

    return steps

def calculate_averages(x, y, threshold):
    avg_before = 0
    before_id = 0
    avg_after = 0
    after_id = 0
    for i in range(0, len(y)):
        if x[i] < threshold:
            avg_before += x[i]
            before_id += 1
        else:
            avg_after += x[i]
            after_id += 1
    if before_id != 0:
        avg_before /= before_id
    if after_id != 0:
        avg_after /= after_id

    return avg_before, avg_after

def build_tree(x, y, depth, MAX_LEAVES):
    if depth <= 0 or len(y) < MAX_LEAVES:
        return Leaf(y)
    depth -= 1

    question = find_best_split(x, y)

    # print('Question: ' + str(question.column) + ' ' + str(question.value))

    true_x, true_y, false_x, false_y = partition(x, y, question)

    if len(true_x) == 0 or len(false_x) == 0:
        return Leaf(y)

    true_branch = build_tree(true_x, true_y, depth, MAX_LEAVES)
    false_branch = build_tree(false_x, false_y, depth, MAX_LEAVES)

    return Decision_Node(question, true_branch, false_branch)

def partition(x, y, question):
    true_x, true_y, false_x, false_y = [], [], [], []
    for i in range(0, len(x)):
        if question.match(x[i]):
            true_x.append(x[i])
            true_y.append(y[i])
        else:
            false_x.append(x[i])
            false_y.append(y[i])
    
    # print(str(len(true_x)) + ' ' + str(len(false_x)))

    return true_x, true_y, false_x, false_y

def find_value(root, data_row):
    if isinstance(root, Leaf):
        return root.value
    if root.question.match(data_row):
        return find_value(root.true_branch, data_row)
    return find_value(root.false_branch, data_row)

def sort_column(x, y, x_id):
    column = [row[x_id] for row in x]
    zipped_lists = zip(column, y)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    column, y_out = [ list(tuple) for tuple in  tuples]

    return column, y_out