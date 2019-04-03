"""
Classify pixels as either vessel or background
"""

import pickle
from sklearn import svm, metrics

def train(feature_image, truth_image):
    """
    Train model from feature array and true classes
    """
    flat_image = feature_image.reshape(-1, feature_image.shape[-1])
    flat_truth = truth_image.flatten()
    model = svm.SVC(kernel='linear')
    model.fit(flat_image, flat_truth) # Train
    pickle.dump(model, open('model.p', 'wb')) # Persist model

def classify(feature_image):
    """
    Classify image from feature array and trained model
    """
    model = pickle.load(open('model.p', 'rb')) # Load model
    shape = feature_image.shape[:2]
    flat_image = feature_image.reshape(-1, feature_image.shape[-1])

    return model.predict(flat_image).reshape(shape)

def assess(truth, prediction):
    """
    Display accuracy of classification
    """
    flat_truth = truth.reshape(-1, truth.shape[-1])
    flat_prediction = prediction.reshape(-1, prediction.shape[-1])
    print('Accuracy: {}'.format(metrics.accuracy_score(flat_truth, flat_prediction)))
