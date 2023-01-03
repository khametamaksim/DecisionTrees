import sys
from id3 import ID3
from node import Node
from dataframe import Dataframe


dataset_parent = sys.argv[1]
dataframe_parent = Dataframe()
dataframe_parent.read_file(dataframe_parent, dataset_parent)

dataset_train = sys.argv[1]
dataframe_train = Dataframe()
dataframe_train.read_file(dataframe_train, dataset_train)

dataset_test = sys.argv[2]
dataframe_test = Dataframe()
dataframe_test.read_file(dataframe_test, dataset_test)

depth_value = 1000
if len(sys.argv) > 3:
    depth_value = int(sys.argv[3])

model = ID3()
root = model.fit(Node(), dataframe_train, dataframe_parent, depth_value)
model.print_branches(root)
predictions = model.print_predictions(root, dataframe_test.rows, dataframe_test.columns)
model.print_model_performance(dataframe_test.classes, predictions)
model.print_confusion_matrix(dataframe_test.classes, predictions)
