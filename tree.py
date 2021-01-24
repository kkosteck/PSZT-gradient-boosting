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
    print(n_features)

    for col in range(n_features):  # for each feature
        current_value, index = best_threshold(x, y) 
        question = Question(col, x[index][col])
        if current_value < best_value: 
            best_value = current_value
            best_question = question

    return best_question

def best_threshold(x, y):
    min_sum = float("inf")
    index = 0
    for i in range(1, len(x)-1):
        current_sum = 0
        for j in range(0, i):
            current_sum +=y[j]
        if current_sum < min_sum: 
            min_sum = current_sum
            index = i
    return min_sum, index

def build_tree(x, y, depth):

    question = find_best_split(x, y)

    if depth == 0:
        return Leaf(y)
    depth -= 1

    true_x, true_y, false_x, false_y = partition(x, y, question)

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