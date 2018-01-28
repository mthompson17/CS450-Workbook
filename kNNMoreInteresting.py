# 1) Reading the sets into dataframes
# 2) Arrays
# 3) Pick randomly
# 4) KFold cross validation
# 5) KNeighborsClassifier: KNN algorithm for training a discrete model
# 6) KNeighborsRegressor: KNN algorithm for training a continuous model
# 7) accuracy_score: compute accuracy of predicted test targets
# 8) Read sets easier
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from pandas import read_csv

# Load the car data from ICS
def loadCar():
    df = pd.io.parsers.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
        header=None)

    # Name each column for dictionary purposes
    df.columns=["buying", "maint", "doors", "persons", "lug_boot", "safety", "accept"]

    # Just get the objects into the dataframe
    obj_df = df.select_dtypes(include=['object']).copy()

    # Dictionary for replacing words with values
    numericalCategories = {"buying":   {"vhigh": 10, "high": 7, "med": 4, "low": 0},
                           "maint":    {"vhigh": 10, "high": 7, "med": 4, "low": 0},
                           "doors":    {"2": 2, "3": 3, "4": 4, "5more": 5},
                           "persons":  {"2": 2, "4": 4, "more": 5},
                           "lug_boot": {"small": 0, "med": 5, "big":10},
                           "safety":   {"low": 0, "med":5, "high":10},
                           "accept":   {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}}

    # Replacing the words with values and turning the df into a list
    obj_df.replace(numericalCategories, inplace=True)
    carData = obj_df.as_matrix()

    # Getting the targets out of the list
    carTargets = []
    for x in carData:
        carTargets.append(x[6])

    # Deleting the targets so the list is just data
    carData = np.delete(carData, 6, 1)
    carTargets = np.array(carTargets)

    return carData, carTargets

# Load pima data from ICS
def loadPima():
    pima = read_csv('pima.csv', header=None)

    # Replace zeroes in columns where it's invalid
    pima[[1,2,3,4,5,6,7]] = pima[[1,2,3,4,5,6,7]].replace(0, np.NaN)
    pima.dropna(inplace=True)

    # Turn it into list, extract targets and data from list
    pimaList = pima.as_matrix()
    pimaTargets = []
    for x in pimaList:
        pimaTargets.append(x[8])
    pimaData = np.delete(pimaList, 8, 1)
    pimaTargets = np.array(pimaTargets)

    return pimaData, pimaTargets

# Load MPG data from ICS
def loadMPG():
    # Set ? values to NaN, and remove NaN
    mpg = read_csv('mpg.csv', header=None, delim_whitespace=True, na_values=['?'])
    mpg.dropna(inplace=True)

    # Turn it into list, extract targets and data from list
    mpgList = mpg.as_matrix()
    mpgTargets = []
    for x in mpgList:
        mpgTargets.append((x[0]))
    mpgData = np.delete(mpgList, 0, 1)
    mpgData = np.delete(mpgList, 8, 1)

    mpgTargets = np.array(mpgTargets)

    return mpgData, mpgTargets

# Use the k-Nearest-Neighbors algorithm to
# accurately train a model based on the data
# and targets
def createModel(data_train, targets_train):
    classifier = KNeighborsClassifier(n_neighbors=10)
    model = classifier.fit(data_train, targets_train)
    return model

# Same as createModel, but uses regressor for MPG data
def createModel2(data_train, targets_train):
    regressor = KNeighborsRegressor(n_neighbors=10)
    model = regressor.fit(data_train, targets_train)
    return model

# Use the model to predict the targets
def modelPredict(model, data_test):
    targets_predicted = model.predict(data_test)
    return targets_predicted

# Get the accuracy of the real KNN model
def getAccuracy(targets_predicted, targets_test):
    accuracy = accuracy_score(targets_test, targets_predicted)
    size = len(targets_test)
    matches = int(round(accuracy * size))
    return accuracy, matches

###################################################################################

# Organize distance-target pairs by closest distance
def getKey(item):
    return item[0]

# Class definition for KNNModel
class KNNModel():
    def __init__(self, data_train, targets_train, n_neighbors):
        self.data_train = data_train
        self.targets_train = targets_train
        self.n_neighbors = n_neighbors

    def predict(self, data_test):
        targets_predictions = []
        for i in data_test:
            distances = []
            for j in self.data_train:
                # Euclidean distance across all attributes using scaled data
                distances.append(sum([(k - l) ** 2 for k, l in zip(i, j)]))

            # Join point's distance with its target
            distTarget = list(zip(distances, self.targets_train))
            distTarget.sort(key=getKey)

            # Get the nearest 'k' distances/targets
            nearest = distTarget[:self.n_neighbors]

            # Organize targets
            targets = [0, 0, 0, 0]

            # Target mode will be determined by index with greatest value
            for n in nearest:
                targets[int(n[1])] += 1

            # Handle pluralities and ties
            maximum = max(targets)
            maxIndex = [i for i, j in enumerate(targets) if j == maximum]
            if (len(maxIndex) == 1):
                targets_predictions.append(maxIndex[0])
            else:
                targets_predictions.append(random.choice(maxIndex))
        return np.array(targets_predictions)

    # MPG predictions
    def narrowDown(self, data_test):
        targets_predictions = []
        for i in data_test:
            distances = []
            for j in self.data_train:
                # Euclidean distance across all attributes using scaled data
                distances.append(sum([(k - l) ** 2 for k, l in zip(i, j)]))

            # Join point's distance with its target
            distTarget = list(zip(distances, self.targets_train))
            distTarget.sort(key=getKey)

            # Get the nearest 'k' distances/targets
            nearest = distTarget[:self.n_neighbors]

            # Determine average MPG for estimate
            mpgTotal = 0
            for n in nearest:
                mpgTotal += n[1]
            mpgEstimate = round((mpgTotal / self.n_neighbors), 1)
            targets_predictions.append(mpgEstimate)

        return np.array(targets_predictions)

# Class definition for KNNClassifier
class KNNClassifier():
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, data_train, targets_train):
        return KNNModel(data_train, targets_train, self.n_neighbors)

# Implement a KNN Classifier to train the
# model to select the correct index of iris
def KNNTrain(data_train, targets_train):
    knnClassifier = KNNClassifier(n_neighbors=10)
    knnModel = knnClassifier.fit(data_train, targets_train)
    return knnModel

# Predict the index of iris for each comparison
def KNNModelPredict(knnModel, data_test):
    targetsKC = knnModel.predict(data_test)
    return targetsKC

# Same as KNNModelPredict, but uses narrowDown for MPG data
def KNNModelPredict2(knnModel, data_test):
    targetsKC = knnModel.narrowDown(data_test)
    return targetsKC

# Get accuracy of the KNN Classifier's predictions
def getKNNAccuracy(targetsKC, targets_test):
    accuracy = accuracy_score(targetsKC, targets_test)
    size = len(targets_test)
    matches = int(round(accuracy * size))
    return accuracy, matches, size

############################################################################

# Use KFold cross validation and prepare the target
# training and testing sets
def prepareSets(dataset, targets):
    sklearnKNN_Accuracy = 0
    sklearnKNN_Matches = 0
    myKNN_Accuracy = 0
    myKNN_Matches = 0
    totalSize = 0

    kf = KFold(n_splits=10)
    for train, test in kf.split(dataset):
        # Set the training and testing sets
        data_train, data_test, targets_train, targets_test = dataset[train], dataset[test], targets[train], targets[test]

        # Iterate through sklearn version
        model = createModel(data_train, targets_train)
        targets_predicted = modelPredict(model, data_test)
        accuracy, matchesKNN = getAccuracy(targets_predicted, targets_test)

        # Iterate through my version
        knnModel = KNNTrain(data_train, targets_train)
        targetsKC = KNNModelPredict(knnModel, data_test)
        accuracyKC, matchesKC, size = getKNNAccuracy(targetsKC, targets_test)

        # Gather totals
        sklearnKNN_Accuracy += accuracy
        myKNN_Accuracy += accuracyKC
        sklearnKNN_Matches += matchesKNN
        myKNN_Matches += matchesKC
        totalSize += size

    # Determine totals
    skKNN_Accuracy = (sklearnKNN_Accuracy / kf.get_n_splits())
    KNN_Accuracy = (myKNN_Accuracy / kf.get_n_splits())

    print("k-Nearest-Neighbors Accuracy: {0:.2%}".format(skKNN_Accuracy))
    print("k-Nearest-Neighbors Matches: {} out of {}".format(sklearnKNN_Matches, totalSize))
    print("KNN Classifier Accuracy: {0:.2%}".format(KNN_Accuracy))
    print("KNN Classifier Matches: {} out of {}".format(myKNN_Matches, totalSize))

def prepareMPG(dataset, targets):
    sklearnKNN_Accuracy = 0
    sklearnKNN_Matches = 0
    myKNN_Accuracy = 0
    myKNN_Matches = 0
    totalSize = 0

    kf = KFold(n_splits=10)
    for train, test in kf.split(dataset):
        data_train, data_test, targets_train, targets_test = dataset[train], dataset[test], targets[train], targets[test]

        # Same set up as prepareSets, except it uses createModel2 and
        # KNNModelPredict2, which implements narrowDown for MPG
        model = createModel2(data_train, targets_train)
        targets_predicted = modelPredict(model, data_test)
        accuracy, matchesKNN = getAccuracy(targets_predicted, targets_test)

        knnModel = KNNTrain(data_train, targets_train)
        targetsKC = KNNModelPredict2(knnModel, data_test)
        accuracyKC, matchesKC, size = getKNNAccuracy(targetsKC, targets_test)

        sklearnKNN_Accuracy += accuracy
        myKNN_Accuracy += accuracyKC
        sklearnKNN_Matches += matchesKNN
        myKNN_Matches += matchesKC
        totalSize += size

    skKNN_Accuracy = (sklearnKNN_Accuracy / kf.get_n_splits())
    KNN_Accuracy = (myKNN_Accuracy / kf.get_n_splits())

    print("k-Nearest-Neighbors Accuracy: {0:.2%}".format(skKNN_Accuracy))
    print("k-Nearest-Neighbors Matches: {} out of {}".format(sklearnKNN_Matches, totalSize))
    print("KNN Classifier Accuracy: {0:.2%}".format(KNN_Accuracy))
    print("KNN Classifier Matches: {} out of {}".format(myKNN_Matches, totalSize))

############################################################################

# Driver for functions
def main():
    carData, carTargets = loadCar()
    prepareSets(carData, carTargets)
    pimaData, pimaTargets = loadPima()
    prepareSets(pimaData, pimaTargets)
    mpgData, mpgTargets = loadMPG()
    prepareMPG(mpgData, mpgTargets)

# Run main
if __name__ == '__main__':
    main()