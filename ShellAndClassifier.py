# For arrays
import numpy as np

# Import iris data
from sklearn import datasets

# Import function to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Import Naive Bayes algorithm to train a model
# based on the data and targets
from sklearn.naive_bayes import GaussianNB

# Load the iris data from sklearn
def loadData():
    iris = datasets.load_iris()
    return iris

# Use train-test-split and prepare the target
# training and testing sets
def prepareSets(iris):
    data_train, data_test, targets_train, targets_test = train_test_split(iris.data, iris.target, test_size=0.30)
    return data_train, data_test, targets_train, targets_test

# Use the Naive Bayes algorithm to accurately
# train a model based on the data and targets
def createModel(data_train, targets_train):
    classifier = GaussianNB()
    model = classifier.fit(data_train, targets_train)
    return model

# Use the model to predict the targets
def modelPredict(model, data_test):
    targets_predicted = model.predict(data_test)
    return targets_predicted

# Display the accuracy of the Naive Bayes model
def displayAccuracy(targets_predicted, targets_test):
    matchesNB = 0
    comparisonsNB = 0
    for x,y in zip(targets_predicted, targets_test):
        comparisonsNB += 1
        if (x == y):
            matchesNB += 1

    print("Using Naive Bayes, there were %d matches out of %d comparisons." % (matchesNB, comparisonsNB))
    return matchesNB

# Class definition for HardCodedModel
class HardCodedModel():
    def predict(self, data_test):
        targets = np.zeros(len(data_test), dtype=int)
        return targets

# Class definition for HardCodedClassifier
class HardCodedClassifier():
    def __init__(self):
        self.model = HardCodedModel()

    def fit(self, data_train, targets_train):
        return self.model

# Implement an "algorithm" to train the model to always select the same index of iris
def hardCodedTrain(data_train, targets_train):
    hardCodedClassifier = HardCodedClassifier()
    hardCodedModel = hardCodedClassifier.fit(data_train, targets_train)
    return hardCodedModel

# Predict the same index of iris for each comparisons
def hardCodedModelPredict(hardCodedModel, data_test):
    targetsHC = hardCodedModel.predict(data_test)
    return targetsHC

# Display accuracy of Hard Coded Classifier's predictions
def displayHardCodedAccuracy(targetsHC, targets_test):
    matchesHC = 0
    comparisonsHC = 0
    for x,y in zip(targetsHC, targets_test):
        comparisonsHC += 1
        if (x == y):
            matchesHC += 1

    print ("Using the Hard Coded Classifier, there were %d matches out of %d comparisons." % (matchesHC, comparisonsHC))
    return matchesHC

# Driver for functions
def main():
    iris = loadData()
    data_train, data_test, targets_train, targets_test = prepareSets(iris)
    model = createModel(data_train, targets_train)
    targets_predicted = modelPredict(model, data_test)
    matchesNB = displayAccuracy(targets_predicted, targets_test)

    hardCodedModel = hardCodedTrain(data_train, targets_train)
    targetsHC = hardCodedModelPredict(hardCodedModel, data_test)
    matchesHC = displayHardCodedAccuracy(targetsHC, targets_test)

    print ("The Naive Bayes Classifier was %.2f times more accurate than the Hard Coded Classifier." % (float(matchesNB) / matchesHC))

# Run main
if __name__ == '__main__':
    main()
