import statistics
import timeit

THRESHOLD_STEP = 5

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
    best_question = None 
    n_features = len(x[0])

    for col in range(n_features):  # for each feature
        x_sorted, y_sorted = sort_column(x, y, col)
        current_value, threshold = best_threshold(x_sorted, y_sorted)
        if threshold >= max(x_sorted):
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
    for i in range(1, len(y), int(len(y) / THRESHOLD_STEP)):
        avg_before_threshold = statistics.mean(y[:i])
        avg_after_threshold = statistics.mean(y[i:])
        sum_of_residuals = 0
        for j in range(0, len(y)):
            if j < i:
                sum_of_residuals += ((y[j] - avg_before_threshold) ** 2)
            else:
                sum_of_residuals += ((y[j] - avg_after_threshold) ** 2)

        if sum_of_residuals < lowest_sum: 
            lowest_sum = sum_of_residuals
            threshold = x[i]
    
    return lowest_sum, threshold

def build_tree(x, y, depth, MAX_LEAVES):
    if depth <= 0 or len(y) < MAX_LEAVES:
        return Leaf(y)
    depth -= 1

    question = find_best_split(x, y)

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