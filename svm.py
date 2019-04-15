"""
Support vector machine training, classifying, and assessment
"""

import logging
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from log_execution import log_execution

@log_execution
def train(feature_images, truth_images):
    """
    Train model from feature array and true classes
    """
    feature_images = np.array(feature_images) # Cast from list to numpy array
    flat_image = feature_images.reshape(-1, feature_images.shape[-1])
    flat_truth = np.ravel(truth_images) # One-dimensional truth

    logging.debug('Training model using %d data points', feature_images.size)

    base_estimator = SVC(gamma='auto', probability=True)
    num_estimators = feature_images.shape[0]
    num_samples = np.prod(feature_images.shape[1:3])
    model = BaggingClassifier(base_estimator, n_estimators=num_estimators, max_samples=num_samples)
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
    probabilities = model.predict_proba(flat_image) # Zero to one probabilities
    probabilities = probabilities[:, 1].reshape(shape) # Probability of vessel

    return probabilities

@log_execution
def assess(truth, probabilities):
    """
    Display accuracy of classification
    """
    prediction = np.where(probabilities >= 0.5, True, False).astype(np.bool)
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

def plot_roc(truth, probabilities):
    """
    Plot ROC curve
    """
    flat_probabilities = np.ravel(probabilities)
    flat_truth = np.ravel(truth) # One-dimensional truth
    fpr, tpr, _ = roc_curve(flat_truth, flat_probabilities)
    auc = roc_auc_score(flat_truth, flat_probabilities)

    print('AUC: {}'.format(auc))
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, label='roc')
    plt.show()
