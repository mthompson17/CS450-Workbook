from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np
import random
import math

class Network:
    def __init__(self, layerList, trainData, testData, targets):
        self.neuronList = [Neuron(layerList[i]) for i in range(len(layerList))]
        self.trainData = trainData
        self.testData = testData
        self.targets = targets
        self.layerList = layerList
        self.learningRate=0.3

    # Go back and update weights according to contribution to error
    # Can't get updating to work
    def backprop(self, neuronList, data, target, predicted):
        act = 0
        for layer in range((len(self.layerList) - 1), -1, -1):
            act += sum(self.neuronList[layer].activations)
        for layer in range((len(self.layerList) - 1), -1, -1):
            error = 0
            if layer == len(self.layerList) - 1:
                error = act * (1-act) * (act - predicted)
                neuronList[layer].errors.append(error)
            else:
                weightedSum = []
                for i in range(self.layerList[layer + 1]):
                    weightedSum.append(sum([x*y for x,y in zip(neuronList[layer + 1].errors,neuronList[layer + 1].inputWts)]))
                wtSum = sum(weightedSum)
                error = act * (1-act) * (wtSum)
                neuronList[layer].errors.append(error)
        for layer in range(len(self.layerList) + 1):
            for node in range(self.layerList[layer]):
                #error = neuronList[layer].errors(node % self.layerList[layer])
                error = random.randint(1,50)/100.0
                rate = self.learningRate
                if layer == 0:
                    newWts = list(map(lambda x, y: (x - ((rate) * (error) * (y))), neuronList[layer].inputWts, data))
                    neuronList[layer].inputWts = newWts
                else:
                    actIn = []
                    actIn.append(-1)
                    for n in range(self.layerList[layer - 1]):
                        actIn.append(neuronList[layer - 1].activations)
                    print(actIn)
                    #newWts = list(map(lambda x, y: (x - ((rate) * (error) * (y))), neuronList[layer].inputWts, actIn))
                    #neuronList[layer].inputWts = newWts

    # Feed the inputs and activations forward
    def feedForward(self, neuronList, data, target):
        #print("layerlist: ")
        #print(self.layerList)
        epoch = 0
        for layer in range(len(self.layerList)):
            #print("layer")
            #print(layer)
            if (layer == 0):
                neuronList[layer].calcOutput(data)
            elif (layer != (len(self.layerList) - 1)):
                neuronList[layer].calcOutput(neuronList[layer - 1].activations)
            else:
                #print("done")
                #print(neuronList[layer - 1].activations)
                predicted = np.argmax(neuronList[layer - 1].activations)
                if predicted == target:
                    return predicted
                else:
                    # Back propagate if prediction is wrong
                    epoch += 1
                    if epoch < 100:
                        print(epoch)
                        self.backprop(neuronList, data, target, predicted)
                        self.feedForward(neuronList, data, target)

    # Determine prediction accuracy (was getting low %s)
    def predict(self):
        predictions = []
        for data in range(len(self.testData)):
            predictions.append(self.feedForward(self.neuronList, self.testData[data], self.targets[data]))
        correct = 0
        overall = 0
        for i in range(len(predictions)):
            overall += 1
            if (predictions[i] == self.targets[i]):
                correct += 1
        print("%d/%d correct" % (correct, overall))

# A neuron layer
class Neuron():
    def __init__(self, numNodes):
        self.inputWts = [(random.randint(-99,99)/100.00) for i in range(numNodes)]
        self.numNodes = numNodes
        self.bias = -1.00
        self.biasWt = (random.randint(-99,99)/100.00)
        self.activations = []
        self.errors = []

    # Determine sigmoid output
    def calcOutput(self, instance):
        output = 0.00
        for x,y in zip(instance, self.inputWts):
            output += x*y
            #print("output")
            #print(output)
        output += self.bias * self.biasWt
        #print("output2")
        #print(output)
        self.activations.append(self.sigmoid(output))
        #print("sigmoid")
        #print(self.sigmoid(output))
        return self.sigmoid(output)

    def sigmoid(self, x):
        return (1.00/(1.00 + math.e**(-x)))

def main():
    # Load iris
    np.set_printoptions(suppress=True)
    iris = datasets.load_iris()
    dtrain, dtest, ttrain, ttest = train_test_split(iris.data, iris.target, test_size=0.3)
    dtrain = preprocessing.normalize(dtrain)
    dtest = preprocessing.normalize(dtest)

    # Set up network structure
    irisInputs = 4
    irisOutputs = 3
    irisNN = Network([irisInputs, 5, irisOutputs], dtrain, dtest, ttest)
    irisNN.predict()

    # Load pima
    pima = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data")
    pima.columns = ["preg","glucose","pressure","skin","insulin","bmi","pedigree","age", "class"]
    no_zero = ['glucose', 'pressure', 'skin', 'bmi', 'insulin', 'age']
    for c in no_zero:
        pima[c] = pima[c].replace(0, np.NaN)
        mean = int(pima[c].mean(skipna=True))
        pima[c] = pima[c].replace(np.NaN, mean)
    pimaNP = pima.as_matrix()

    pimaTargets = []
    for x in pimaNP:
        pimaTargets.append(int(x[8]))
    pimaNP = np.delete(pimaNP, 8, 1)
    targetNP = np.array(pimaTargets)

    dtrain2, dtest2, ttrain2, ttest2 = train_test_split(pimaNP, targetNP, test_size=0.3)
    dtrain2 = preprocessing.normalize(dtrain2)
    dtest2 = preprocessing.normalize(dtest2)

    # Set up network structure
    pimaInputs = 8
    pimaOutputs = 2
    pimaNN = Network([pimaInputs, 5, pimaOutputs], dtrain2, dtest2, ttest2)
    pimaNN.predict()

    
if __name__ == '__main__':
    main()