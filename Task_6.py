'''
#######################################################
                        KNN Classification
#######################################################
'''
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets

# The digits dataset
digits = datasets.load_digits()

from sklearn.model_selection import train_test_split
import numpy as np
# Training and testing split,
# 75% for training and 25% for testing
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(digits.data), digits.target, test_size=0.5, random_state=42)

# take 10% of the training data and use that for validation
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1, random_state=84)

# Checking sizes of each data split
print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))


kVals = range(1, 30, 2)
accuracies = []
from sklearn.neighbors import KNeighborsClassifier
# loop over kVals
for k in range(1, 30, 2):
    # train the classifier with the current value of `k`
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData, trainLabels)

    # evaluate the model and print the accuracies list
    score = model.score(valData, valLabels)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)
    
#i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[5],accuracies[5] * 100))

model = KNeighborsClassifier(n_neighbors=kVals[5])
model.fit(trainData, trainLabels)

# Predict labels for the test set
predictions = model.predict(testData)

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(testLabels, predictions)
import numpy as np
print("Number of mislabeled points in KNN :{}".format(np.sum((testLabels != predictions))))
n_samples = len(digits.images)
images_and_predictions_knn = list(zip(digits.images[n_samples // 2:], predictions))
for index_knn, (image_knn, prediction_knn) in enumerate(images_and_predictions_knn[:6]):
    plt.subplot(5, 2, index_knn + 5)
    plt.axis('off')
    plt.imshow(image_knn, cmap=plt.cm.gray_r, interpolation='bicubic')
    plt.title('Prediction: %i' % prediction_knn)
    
    
