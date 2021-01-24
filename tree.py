import statistics

class Leaf:
    def __init__(self, data):
        self.value = statistics.mean(data)

class Decision_Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]

        return val >= self.value

def find_best_split(data):
    best_value = float("inf")
    best_question = None 
    n_features = len(data[0]) - 1

    for col in range(n_features):  # for each feature
        current_value, index = best_threshold(data[col]) 
        question = Question(col, data[col][index])
        if current_value < best_value: 
            best_value = current_value
            best_question = question

    return best_question

def best_threshold(data):
    min_sum = float("inf")
    index = 0
    for i in range(data):
        current_sum = 0
        for j in range(0, i):
            current_sum +=predicted_value[j]
        if current_sum < min_sum: 
            min_sum = current_sum
            index = j
    return min_sum, index

def build_tree(data, depth, predicted_value):
    question = find_best_split(data)

    if depth == 0:
        return Leaf(data)
    depth -= 1

    true_data, false_data = partition(data, question)

    true_branch = build_tree(true_data, depth, predicted_value)

    false_branch = build_tree(false_data, depth, predicted_value)

    return Decision_Node(question, true_branch, false_branch)

def partition(data, question):
    true_data, false_data = [], []
    for row in data:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_data, false_data