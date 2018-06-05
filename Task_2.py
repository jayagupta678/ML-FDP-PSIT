# Part 2 - Visualizing a Decision Tree - https://youtu.be/tNa99PG8hR8

# Build one on a real dataset, add code to visualize it, and practice reading it - so you can see how it works under the
# hood.

# Use Iris flower data set: https://en.wikipedia.org/wiki/Iris_flower_data_set
# Identify type of flower based on measurements
# Dataset includes 3 species of Iris flowers: setosa, versicolor, virginica
# 4 features used to describe: length and width of sepal and petal
# 50 examples of each type for 150 total examples
'''
# Goals
# 1-Import dataset
# 2-Train a classifier
# 3-Predict label for new flower
# 4-Visualize the tree
'''
# scikit-learn datasets: http://scikit-learn.org/stable/datasets/
# already includes Iris dataset: load_iris

from sklearn.datasets import load_iris

iris = load_iris()

print(iris.feature_names)  # metadata: names of the features
print(iris.target_names)  # metadata: names of the different types of flowers

# print iris.data  # features and examples themselves
print (iris.data[0])  # first flower
print (iris.target[0])  # contains the labels

#Python 3.6.0 |Anaconda 4.3.1

for i in range(len(iris.target)):
    print( "Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))

import numpy as np
from sklearn import tree

# here, we remove the first example of each flower
# found at indices: 0, 50, 100
test_idx = [0, 50, 100]

# create 2 new sets of variables, for training and testing
# training data
# remove the entires from the data and target variables
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# create new classifier
clf = tree.DecisionTreeClassifier()
# train on training data
clf.fit(train_data, train_target)

# what we expect
print(test_target)

# what tree predicts
print(clf.predict(test_data))

'''
#######################################
Decision Tree Visualization of data
######################################
'''
import pydotplus
import collections

dot_data = tree.export_graphviz(clf,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
 
colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)
 
for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))
 
for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])
 
graph.write_png('123.png')