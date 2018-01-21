# 1) Arrays
# 2) Pick randomly
# 3) Random number generator
# 4) datasets: iris data
# 5) train-test-split: split data into training and testing sets
# 6) KNeighborsClassifier: KNN algorithm for training a model
# 7) accuracy_score: compute accuracy of predicted test targets
import numpy as np
import random
from random import *
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the iris data from sklearn
def loadData():
    iris = datasets.load_iris()
    return iris

# Separate, scale, and recombine the iris data
def scaleIris(iris):
    # Make a list for each attribute
    sl = [] # sepal length
    sw = [] # sepal width
    pl = [] # petal length
    pw = [] # petal width

    # Put each attribute's data in its own array
    for x in iris.data:
        sl.append(x[0])
        sw.append(x[1])
        pl.append(x[2])
        pw.append(x[3])

    # Calculate the range of each attribute
    rangeSL = max(sl) - min(sl)
    rangeSW = max(sw) - min(sw)
    rangePL = max(pl) - min(pl)
    rangePW = max(pw) - min(pw)

    # Scaled lists
    slScaled = []
    swScaled = []
    plScaled = []
    pwScaled = []

    # Make the scaled lists for each attribute
    for x in sl:
        slScaled.append((x - min(sl)) / rangeSL)
    for x in sw:
        swScaled.append((x - min(sw)) / rangeSW)
    for x in pl:
        plScaled.append((x - min(pl)) / rangePL)
    for x in pw:
        pwScaled.append((x - min(pw)) / rangePW)

    # Realign the scaled data into a correctly-shaped array (same as iris.data)
    scaled_iris_data = []
    for w, x, y, z in zip(slScaled, swScaled, plScaled, pwScaled):
        scaled_iris_data.append([w, x, y, z])

    return scaled_iris_data

# Use train-test-split and prepare the target
# training and testing sets
def prepareSets(iris, scaled_iris_data): # using scaled data
    data_train, data_test, targets_train, targets_test = train_test_split(scaled_iris_data, iris.target, test_size=0.30)
    return data_train, data_test, targets_train, targets_test

# Use the k-Nearest-Neighbors algorithm to
# accurately train a model based on the data
# and targets
def createModel(data_train, targets_train):
    classifier = KNeighborsClassifier(n_neighbors=3)
    model = classifier.fit(data_train, targets_train)
    return model

# Use the model to predict the targets
def modelPredict(model, data_test):
    targets_predicted = model.predict(data_test)
    return targets_predicted

# Display the accuracy of the real KNN model
def displayAccuracy(targets_predicted, targets_test):
    accuracy = accuracy_score(targets_test, targets_predicted)
    size = len(targets_test)
    matches = int(accuracy * size)
    print("k-Nearest-Neighbors Accuracy: {0:.2%}".format(accuracy))
    print("k-Nearest-Neighbors Matches: {} out of {}".format(matches, size))
    return matches

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
                distances.append(sum([(k - l)**2 for k,l in zip(i, j)]))

            # Join point's distance with its target
            distTarget = list(zip(distances, self.targets_train))
            distTarget.sort(key=getKey)

            # Get the nearest 'k' distances/targets
            nearest = distTarget[:self.n_neighbors]

            # Organize targets
            target0, target1, target2 = 0,0,0
            for n in nearest:
                if n[1] == 0:
                    target0 += 1
                elif n[1] == 1:
                    target1 += 1
                else:
                    target2 += 1

            # Handle pluralities and ties
            if (target0 > target1 and target0 > target2):     # 0 has most
                targets_predictions.append(0)
            elif (target1 > target0 and target1 > target2):   # 1 has most
                targets_predictions.append(1)
            elif (target2 > target0 and target2 > target1):   # 2 has most
                targets_predictions.append(2)
            elif (target0 == target1 and target1 == target2): # all equal
                targets_predictions.append(randint(0, 2))
            elif (target0 == target1):                        # 0 and 1 equal
                targets_predictions.append(randint(0, 1))
            elif (target1 == target2):                        # 1 and 2 equal
                targets_predictions.append(randint(1, 2))
            elif (target0 == target2):                        # 0 and 2 equal
                targets_predictions.append(random.choice([0, 2]))
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
    knnClassifier = KNNClassifier(n_neighbors=3)
    knnModel = knnClassifier.fit(data_train, targets_train)
    return knnModel

# Predict the index of iris for each comparison
def KNNModelPredict(knnModel, data_test):
    targetsKC = knnModel.predict(data_test)
    return targetsKC

# Display accuracy of the KNN Classifier's predictions
def displayKNNAccuracy(targetsKC, targets_test):
    accuracy = accuracy_score(targetsKC, targets_test)
    size = len(targets_test)
    matches = int(accuracy * size)
    print("KNN Classifier Accuracy: {0:.2%}".format(accuracy))
    print("KNN Classifier Matches: {} out of {}".format(matches, size))
    return matches

###############################################################################

# Driver for functions
def main():
    iris = loadData()
    scaled_iris_data = scaleIris(iris)
    data_train, data_test, targets_train, targets_test = prepareSets(iris, scaled_iris_data)
    model = createModel(data_train, targets_train)
    targets_predicted = modelPredict(model, data_test)
    matchesKNN = displayAccuracy(targets_predicted, targets_test)

    knnModel = KNNTrain(data_train, targets_train)
    targetsKC = KNNModelPredict(knnModel, data_test)
    matchesKC = displayKNNAccuracy(targetsKC, targets_test)

    # Handle different comparison cases
    if (matchesKC < matchesKNN):
        print("\nk-Nearest-Neighbors was %.2f times more accurate than the KNN Classifier." % (float(matchesKNN) / matchesKC))
    elif (matchesKC > matchesKNN):
            print("\nThe KNN Classifier was %.2f times more accurate than k-Nearest-Neighbors." % (float(matchesKC) / matchesKNN))
    else:
        print("\nThe KNN Classifier was as accurate as k-Nearest-Neighbors.")

# Run main
if __name__ == '__main__':
    main()