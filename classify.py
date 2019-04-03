"""
Classify pixels as either vessel or background
"""

from sklearn import svm, metrics

def train(feature_image, truth_image):
    """
    Train model from feature array and true classes
    """
    flat_image = feature_image.reshape(-1, feature_image.shape[-1])
    flat_truth = truth_image.flatten()
    print(feature_image.shape, flat_image.shape, flat_truth.shape)
    clf = svm.SVC(kernel='linear')
    clf.fit(flat_image, flat_truth)

    return clf

def classify(feature_image, model):
    """
    Classify image from feature array and trained model
    """
    shape = feature_image.shape[:2]
    flat_image = feature_image.reshape(-1, feature_image.shape[-1])

    return model.predict(flat_image).reshape(shape)

def assess(truth, prediction):
    """
    Display accuracy of classification
    """
    flat_truth = truth.reshape(-1, truth.shape[-1])
    flat_prediction = prediction.reshape(-1, prediction.shape[-1])
    print(flat_prediction.shape, flat_truth.shape)
    print('Accuracy: {}'.format(metrics.accuracy_score(flat_truth, flat_prediction)))
