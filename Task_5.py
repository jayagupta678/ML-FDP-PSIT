# Let's Write a Pipeline - Machine Learning Recipes #4 - https://youtu.be/84gqSbLcBFE

# How to test a model and determine accuracy

# Partition data into 2 sets, train and test

# import a dataset
from sklearn import datasets

iris = datasets.load_iris()

# Can think of classifier as a function f(x) = y
X = iris.data  # features
y = iris.target  # labels

# partition into training and testing sets
from sklearn.model_selection import train_test_split

# test_size=0.5 -> split in half
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

'''#####################################
         Decision Tree Classifier
   #####################################'''
from sklearn import tree

my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)

# predict
predictions = my_classifier.predict(X_test)
print(predictions)
print(y_test)
# test
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))

'''#####################################
         KNN Classifier
   #####################################'''

from sklearn.neighbors import KNeighborsClassifier

my_classifier = KNeighborsClassifier()
my_classifier.fit(X_train, y_train)

# predict
predictions = my_classifier.predict(X_test)
print(predictions)

# test
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))

'''#####################################
         Gaussian Naive Bayes Classifier
   #####################################'''

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)
predictions = gnb.predict(X_test)
print(predictions)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))



'''
#######################################################
                SVM Classification
#######################################################
'''
# Standard scientific Python imports
#import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm
clf = svm.SVC(gamma=0.001)
clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(predictions)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))

