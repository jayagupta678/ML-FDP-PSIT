"""
================================
Recognizing hand-written digits
================================
An example showing how the scikit-learn can be used to recognize images of
hand-written digits.
"""
'''
#######################################################
                SVM Classification
#######################################################
'''
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm

# The digits dataset
digits = datasets.load_digits()

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:6]):
    # index is a column, (image, label) is a tuple
    # image intensities are given in an array and labels are the values of digit
    plt.subplot(3,3 , index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='bicubic')
    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
#data = digits.images.reshape((n_samples, -1))
data = digits.data

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])
'''
confusion matrix is used to evaluate 
the quality of the output of a classifier.
The diagonal elements represent the number of points
for which the predicted label is equal to the true label, 
while off-diagonal elements are those that are mislabeled 
by the classifier. The higher the diagonal values
of the confusion matrix the better, 
indicating many correct predictions.
'''

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(expected, predicted)
import numpy as np
print("Number of mislabeled points in SVM :{}".format(np.sum((expected != predicted))))

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:6]):
    plt.subplot(3, 3, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='bicubic')
    plt.title('Prediction: %i' % prediction)

print(digits.images.shape) 
'''
#######################################################
                        KNN Classification
#######################################################
'''

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
    
    
