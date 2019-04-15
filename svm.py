"""
Support vector machine training, classifying, and assessment
"""

import logging
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot
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
    prediction = np.where(probabilities >= 0.5, True, False) # Binary classification

    return probabilities, prediction

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
    logging.info('Sensitivity: %f', sensitivity)
    logging.info('Specificity: %f', specificity)
    logging.info('Accuracy: %f', accuracy)

def plot_roc(truth, probabilities):
    """
    Plot ROC curve
    """
    flat_probabilities = np.ravel(probabilities)
    flat_truth = np.ravel(truth) # One-dimensional truth
    fp_rate, tp_rate, _ = roc_curve(flat_truth, flat_probabilities)
    auc = roc_auc_score(flat_truth, flat_probabilities)

    logging.info('AUC: %f', auc)

    logging.getLogger('matplotlib').setLevel(logging.ERROR) # Mute matplotlib
    pyplot.figure()
    pyplot.ylabel('True Positive Rate')
    pyplot.xlabel('False Positive Rate')
    pyplot.title('Receiver Operating Characteristic')
    pyplot.plot(fp_rate, tp_rate, label='AUC: {}'.format(auc))
    pyplot.legend(loc='lower right')
    pyplot.show()
