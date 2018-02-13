# 1) Arrays
# 2) iris data
# 3) train-test-split: split data into training and testing sets
# 4) Off the shelf DTC
# 5) accuracy_score: compute accuracy of predicted test targets
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the iris data from sklearn
def loadData():
    iris = datasets.load_iris()
    discretized = [["" for i in range(len(iris.data[0]))] for j in range(len(iris.data))]
    print(discretized)
    print(iris.data)
    for index, x in enumerate(iris.data):
        if (x[0] < 5.5):
            discretized[index][0] = 0
        elif (x[0] < 6.3):
            discretized[index][0] = 1
        else:
            discretized[index][0] = 2

        if (x[1] < 3.0):
            discretized[index][1] = 0
        elif (x[1] < 3.3):
            discretized[index][1] = 1
        else:
            discretized[index][1] = 2

        if (x[2] < 3.0):
            discretized[index][2] = 0
        elif (x[2] < 4.9):
            discretized[index][2] = 1
        else:
            discretized[index][2] = 2

        if (x[3] < 1.0):
            discretized[index][3] = 0
        elif (x[3] < 1.6):
            discretized[index][3] = 1
        else:
            discretized[index][3] = 2

        print(discretized[index])
    return discretized, iris

# Use train-test-split and prepare the target
# training and testing sets
def prepareSets(discretized, iris):
    data_train, data_test, targets_train, targets_test = train_test_split(discretized, iris.target, test_size=0.30)
    data_train = np.array(data_train)
    data_test = np.array(data_test)
    targets_train = np.array(targets_train)
    targets_test = np.array(targets_test)
    return data_train, data_test, targets_train, targets_test

# Use the k-Nearest-Neighbors algorithm to
# accurately train a model based on the data
# and targets
def createModel(data_train, targets_train):
    classifier = DecisionTreeClassifier()
    model = classifier.fit(data_train, targets_train)
    return model

# Use the model to pr edict the targets
def modelPredict(model, data_test):
    targets_predicted = model.predict(data_test)
    return targets_predicted

# Display the accuracy of the real KNN model
def displayAccuracy(targets_predicted, targets_test):
    accuracy = accuracy_score(targets_test, targets_predicted)
    size = len(targets_test)
    matches = int(accuracy * size)
    print("Decision Tree Accuracy: {0:.2%}".format(accuracy))
    print("Decision Tree Matches: {} out of {}".format(matches, size))
    return matches

###################################################################################

# Organize distance-target pairs by closest distance
def getKey(item):
    return item[0]

class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = []
        self.value = ""

    def __str__(self):
        return self.attribute

# Class definition for Decision Tree
class DecisionTree():
    def __init__(self, data_train, targets_train):
        self.data_train = data_train
        self.targets_train = targets_train

    def subsets(self, data_train, targets_train, shouldDel):
        dictionary = {}

        # Unique items
        items = np.unique(data_train[:, targets_train])
        items.reshape(1, -1)
        print("Items1: ", items)

        count = np.zeros((items.shape[0], 1), int)
        count.reshape(1, -1)

        # Get counts of unique items
        for x in range(items.shape[0]):
            for y in range(data_train.shape[0]):
                if data_train[y, targets_train] == items[x]:
                    count[x] += 1

        # Adding to dictionary and adjusting position of items
        for x in range(items.shape[0]):
            dictionary[items[x]] = np.empty((count[x], data_train.shape[1]), dtype="S32")
            position = 0
            for y in range(data_train.shape[0]):
                if data_train[y, targets_train] == items[x]:
                    dictionary[items[x]][position] = data_train[y]
                    position += 1
            if shouldDel:
                dictionary[items[x]] = np.delete(dictionary[items[x]], targets_train, 1)

        print ("Items2: ", items)
        print ("Dictionary: ", dictionary)

        return items, dictionary

    # Calculate entropy of system and attributes
    def calcEntropy(self, data):
        items = np.unique(data)
        print ("ItemsE: ", items)
        if items.size == 1:
            return 0

        counts = np.zeros(items.shape[0], 1)
        sums = 0

        # Determine count of the item we're examining
        for x in range(items.shape[0]):
            counts[x] = sum(data == items[x]) / (data.size * 1.0)
        print ("Counts: ",counts)

        # Formula
        for count in counts:
            sums += -1 * count * np.log2(count)
        print ("Sums: ", sums)
        return sums

    # Determine infomation gained
    def infoGain(self, data_train, targets_train):
        items, dictionary = self.subsets(data_train, targets_train, False)

        # Initialize
        size = data_train.shape[0]
        entropies = np.zeros(items.shape[0], 1)
        vals = np.zeros(items.shape[0], 1)

        # Establish the entropies and ratios of items
        for x in range(items.shape[0]):
            ratio = dictionary[items[x]].shape[0]/(float(size))
            entropies[x] = ratio * self.calcEntropy(dictionary[items[x]][:, -1])
            vals[x] = ratio * np.log2(ratio)

        print("Ratio: ", ratio)
        print("Entropies: ", entropies)
        print("Vals: ", vals)

        # Total entropy of system
        totalE = self.calcEntropy(data_train[:, -1])
        info = -1 * sum(vals)

        print("TotalE1: ", totalE)
        for x in range(entropies.shape[0]):
            totalE -= entropies[x]

        print("TotalE2: ", totalE)
        return totalE - info

    # Recursively build tree
    def create_node(self, data_train, meta):
        # If only one class left
        if (np.unique(data_train[:])).shape[0] == 1:
            node = Node("")
            node.value = np.unique(data_train[:])[0]
            return node

        infoGains = np.zeros((data_train.shape[1] - 1, 1))

        # Establish info gains
        for col in range(data_train.shape[1] - 1):
            infoGains[col] = self.infoGain(data_train, col)
        print ("IG: ", infoGains)
        split = np.argmax(infoGains)

        # Initialize node
        node = Node(meta[split])
        meta = np.delete(meta, split, 0)

        # Get subset after the split
        items, dictionary = self.subsets(data_train, split, True)

        # Create children
        for x in range(items.shape[0]):
            child = self.create_node(dict[items[x]], meta)
            node.children.append((items[x], child))

        return node

    def empty(size):
        s = ""
        for x in range(size):
            s += "   "
        return s

    # Print out tree
    def print_tree(self, node, level):
        if node.value != "":
            print(self.empty(level), node.value)
            return

        print(self.empty(level), node.attribute)

        for value, n in node.children:
            print(self.empty(level + 1), value)
            self.print_tree(n, level + 2)

    def predict(self, data_test):
        targets_predictions = ["?????"]
        return np.array(targets_predictions)

# Class definition for Decision Tree Classifier
class DTreeClassifier():
    def fit(self, data_train, targets_train):
        return DecisionTree(data_train, targets_train)

# Implement a Decision Tree to train the
# model to select the correct index of iris
def DTreeTrain(data_train, targets_train):
    DTClassifier = DTreeClassifier()
    DTModel = DTClassifier.fit(data_train, targets_train)
    tree = DTModel.create_node(data_train, "")
    return tree, DTModel

# Predict the index of iris for each comparison
def DTModelPredict(DTModel, data_test, tree):
    targetsDT = DTModel.predict(tree, data_test)
    return targetsDT

# Display accuracy of the KNN Classifier's predictions
def displayDTAccuracy(targetsDT, targets_test):
    accuracy = accuracy_score(targetsDT, targets_test)
    size = len(targets_test)
    matches = int(accuracy * size)
    print("DT Classifier Accuracy: {0:.2%}".format(accuracy))
    print("DT Classifier Matches: {} out of {}".format(matches, size))
    return matches

###############################################################################

# Driver for functions
def main():
    discretized, iris = loadData()
    data_train, data_test, targets_train, targets_test = prepareSets(discretized, iris)
    model = createModel(data_train, targets_train)
    targets_predicted = modelPredict(model, data_test)
    matchesTree = displayAccuracy(targets_predicted, targets_test)

    DTModel = DTreeTrain(data_train, targets_train)
    targetsDT = DTModelPredict(DTModel, data_test)
    matchesDT = displayDTAccuracy(targetsDT, targets_test)

    # Handle different comparison cases
    if (matchesDT < matchesTree):
        print("\nThe Decision Tree Classifier was %.2f times more accurate than the DT Classifier." % (float(matchesKNN) / matchesKC))
    elif (matchesDT > matchesTree):
            print("\nThe DT Classifier was %.2f times more accurate than the Decision Tree Classifier." % (float(matchesKC) / matchesKNN))
    else:
        print("\nThe DT Classifier was as accurate as the Decision Tree Classifier.")

# Run main
if __name__ == '__main__':
    main()