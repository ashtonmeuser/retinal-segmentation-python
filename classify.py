"""
Classify pixels as either vessel or background
"""

from sklearn import svm

def classify():
    """
    """
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = svm.SVC(gamma='scale')
    clf.fit(X, y)
