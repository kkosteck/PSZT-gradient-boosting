import statistics

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

        return val >= self.value

def find_best_split(x, y):
    best_value = float("inf")
    best_question = None 
    n_features = len(x[0])

    for col in range(n_features):  # for each feature
        current_value, index = best_threshold(x, y) 
        question = Question(col, x[index][col])
        if current_value < best_value: 
            best_value = current_value
            best_question = question

    return best_question

def best_threshold(x, y):
    lowest_sum = float('inf')
    index = None
    for i in range(1, len(x)):
        avg_before_threshold = statistics.mean(y[:i])
        avg_after_threshold = statistics.mean(y[i:])
        sum_of_residuals = 0
        for j in range(0, len(x)):
            if j < i:
                sum_of_residuals += ((y[j] - avg_before_threshold) ** 2)
            else:
                sum_of_residuals += ((y[j] - avg_after_threshold) ** 2)

        if sum_of_residuals < lowest_sum: 
            lowest_sum = sum_of_residuals
            index = i
    
    return lowest_sum, index




def build_tree(x, y, depth):
    if depth == 0 or len(y) < 2:
        return Leaf(y)
    depth -= 1

    question = find_best_split(x, y)
    true_x, true_y, false_x, false_y = partition(x, y, question)
    if not true_x or not false_x:
        return Leaf(y)

    true_branch = build_tree(true_x, true_y, depth)
    false_branch = build_tree(false_x, false_y, depth)

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
    return true_x, true_y, false_x, false_y

def find_value(root, data_row):
    if isinstance(root, Leaf):
        return root.value
    if root.question.match(data_row):
        return find_value(root.true_branch, data_row)
    return find_value(root.false_branch, data_row)