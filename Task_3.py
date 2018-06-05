'''
#######################################
            KNN Classifier
######################################
'''
import numpy as np
from sklearn.datasets import load_iris, load_diabetes

iris = load_diabetes()
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


from sklearn.neighbors import KNeighborsClassifier

'''#defalt value for k is '5' here'''

my_classifier = KNeighborsClassifier()
my_classifier.fit(train_data, train_target)

# what we expect
print(test_target)

# what KNN predicts
predictions = my_classifier.predict(test_data)
print(predictions)
