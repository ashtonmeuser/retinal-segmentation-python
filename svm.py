"""
Support vector machine training, classifying, and assessment
"""

import pickle
import numpy as np
from sklearn import svm
from log_execution import log_execution

@log_execution
def train(feature_image, truth_image): # TODO: Accept list of feature_images
    """
    Train model from feature array and true classes
    """
    flat_image = feature_image.reshape(-1, feature_image.shape[-1])
    flat_truth = truth_image.flatten()
    model = svm.SVC(gamma='auto')
    model.fit(flat_image, flat_truth) # Train
    pickle.dump(model, open('model.p', 'wb')) # Persist model

@log_execution
def classify(feature_image):
    """
    Classify image from feature array and trained model
    """
    model = pickle.load(open('model.p', 'rb')) # Load model
    shape = feature_image.shape[:2]
    flat_image = feature_image.reshape(-1, feature_image.shape[-1])

    return model.predict(flat_image).reshape(shape)

@log_execution
def assess(truth, prediction):
    """
    Display accuracy of classification
    """
    true_positive = np.count_nonzero(np.logical_and(truth, prediction))
    true_negative = np.count_nonzero(np.logical_and(~truth, ~prediction))
    false_positive = np.count_nonzero(np.logical_and(~truth, prediction))
    false_negative = np.count_nonzero(np.logical_and(truth, ~prediction))
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive +
                                                  false_negative)
    print('Sensitivity: {}'.format(sensitivity))
    print('Specificity: {}'.format(specificity))
    print('Accuracy: {}'.format(accuracy))
