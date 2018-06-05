================================
Recognizing hand-written digits
================================
An example showing how the scikit-learn can be used to recognize images of
hand-written digits.

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
