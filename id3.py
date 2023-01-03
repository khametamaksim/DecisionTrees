import math

from node import Node
from dataframe import Dataframe

branch = []
branches = []


class ID3:
    def calculate_entropy(self, classes):
        rows = len(classes)
        entropy = 0

        for c in set(classes):
            total_class_count = classes.count(c)
            total_class_entropy = -(total_class_count / rows) * math.log(total_class_count / rows, 2)
            entropy += total_class_entropy
        return entropy

    def calculate_info_gain(self, feature, dataframe):
        feature_info = 0.0
        feature_values = []
        feature_index = dataframe.columns.index(feature)
        for i in range(len(dataframe.rows)):
            feature_values.append(dataframe.rows[i][feature_index])
        unique_feature_values = list(set(feature_values))

        for feature_value in unique_feature_values:
            feature_rows = []
            classes = []
            for row in dataframe.rows:
                if feature_value in row[feature_index]:
                    feature_rows.append(row)
                    classes.append(dataframe.classes[dataframe.rows.index(row)])

            feature_value_entropy = self.calculate_entropy(classes)
            feature_value_probability = len(feature_rows) / len(dataframe.rows)
            feature_info += feature_value_probability * feature_value_entropy

        info_gain = self.calculate_entropy(dataframe.classes) - feature_info
        return info_gain

    def find_most_informative_feature(self, dataframe):
        max_info_gain = -1
        max_info_feature = "zyx"

        for feature in dataframe.columns:
            feature_info_gain = self.calculate_info_gain(feature, dataframe)
            #print("IG(" + feature + ")=" + str(feature_info_gain))
            if max_info_gain < feature_info_gain or (max_info_gain == feature_info_gain and max_info_feature > feature):
                max_info_gain = feature_info_gain
                max_info_feature = feature
        return max_info_feature

    def calculate_max_frequency(self, D):
        max_frequency_value = 0
        max_frequency_class = "zyx"

        for c in D.classes:
            frequency_value = D.classes.count(c)
            if max_frequency_value < frequency_value or (max_frequency_value == frequency_value and max_frequency_class > c):
                max_frequency_value = frequency_value
                max_frequency_class = c

        return max_frequency_class

    def fit(self, node, D, D_parent, depth):


        if not D.columns or len(set(D.classes)) == 1 or depth == 0:
            node.leaf = True
            node.decision = self.calculate_max_frequency(D)
            return node

        if not all(D.rows):
            node.leaf = True
            node.decision = self.calculate_max_frequency(D_parent)
            return node

        feature = self.find_most_informative_feature(D)
        feature_index = D.columns.index(feature)

        columns = D.columns.copy()
        columns.remove(feature)

        feature_values = []
        for i in range(len(D.rows)):
            feature_values.append(D.rows[i][feature_index])
        unique_feature_values = set(feature_values)

        for value in unique_feature_values:
            new_node = Node()
            new_node.feature = feature
            new_node.value = value
            new_rows = []
            new_classes = []

            for row in D.rows:
                if row[feature_index] == value:
                    new_row = row.copy()
                    del new_row[feature_index]
                    new_rows.append(new_row)
                    new_classes.append(D.classes[D.rows.index(row)])

            D_new = Dataframe()
            D_new.columns = columns
            D_new.rows = new_rows
            D_new.label = D.label
            D_new.classes = new_classes

            next_node = self.fit(Node(), D_new, D, depth - 1)

            new_node.children.append(next_node)
            node.children.append(new_node)

        return node

    def build_branches(self, root):

        global branch
        global branches

        if root.value:
            branch.append(str(len(branch) + 1) + ":" + root.feature + "=" + root.value)

        if len(root.children) == 0:
            branch.append(root.decision)
            branches.append(branch.copy())
            branch.pop()
            return

        for i in range(len(root.children)):
            self.build_branches(root.children[i])

        if branch and root.value:
            branch.pop()

    def print_branches(self, root):

        global branches

        self.build_branches(root)

        print("[BRANCHES]: ")

        branches.sort(key=len)
        for b in branches:
            for node in b:
                print(node, end=" ")
            print()

    def predict(self, root, row, columns):

        columns = columns.copy()
        value_row = row.copy()

        if root.leaf:
            return root.decision

        root_feature = root.children[0].feature
        for i in range(len(columns)):
            if columns[i] == root_feature:
                value = row[i]
                children_values = []
                for j in range(len(root.children)):
                    children_values.append(root.children[j].value)
                    if root.children[j].value == value:
                        del value_row[i]
                        columns.remove(columns[i])
                        return self.predict(root.children[j].children[0], value_row, columns)
                if value not in children_values:
                    decisions = []
                    for j in range(len(children_values)):
                        value_row[i] = children_values[j]
                        decisions.append(self.predict(root.children[j].children[0], value_row, columns))
                    return max(sorted(set(decisions)), key=decisions.count)

    def print_predictions(self, root, rows, columns):
        predictions = ["[PREDICTIONS]:"]
        for row in rows:
            predictions.append(self.predict(root, row, columns))

        for prediction in predictions:
            print(prediction, end=" ")
        print()
        return predictions[1:]

    def print_model_performance(self, expected, predictions):
        correct = 0
        for i in range(len(expected)):
            if expected[i] == predictions[i]:
                correct += 1
        accuracy = correct / len(expected)
        print("[ACCURACY]: %.5f" % accuracy)

    def print_confusion_matrix(self, expected, predictions):
        classes = sorted(set(expected))
        scale = len(classes)
        matrix = [[0]*scale for i in range(scale)]

        print("[CONFUSION_MATRIX]: ")
        for i in range(scale):
            for j in range(scale):
                for n in range(len(expected)):
                    if classes[i] == expected[n] and classes[j] == predictions[n]:
                        matrix[i][j] += 1
            print(' '.join(map(str, matrix[i])))
