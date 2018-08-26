import importlib, buildTreeC45, os, sys, time, math
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import copy
from xmlToDict import xmlToDict
from plotTree import plotTree


# required to plot the graph of decision tree
os.environ["PATH"] += os.pathsep + 'C:/graphviz-2.38/release/bin/'


# function to check if a particular package exists on the system
def checkPackage(packageName):
    checkPackage = importlib.util.find_spec(packageName)
    return checkPackage is not None


#function to read data and split it into training and testing sets
# uses random.rand from numpy to split data randomly
def read_data(dataFileName, namesFileName):
    namesFile = open(namesFileName, "r")
    columns = namesFile.read().split(",")
    dataSet = pd.read_csv(dataFileName, names = columns)
    splitDataSet = np.random.rand(len(dataSet)) < 0.8
    trainData = dataSet[splitDataSet]
    testData = dataSet[~splitDataSet]
    return trainData, testData


#implements k-fold cross validation for k = 10
def kFoldValidation(dataset):
    K=10
    # Stores accuracy of the 10 runs
    accuracy = []
    start = time.clock()

    for k in range(K):
        print("Doing fold ", k)
        training_set = copy.deepcopy(dataset)
        test_set = copy.deepcopy(dataset)
        i = 0
        for index, row in dataset.iterrows():
            if i % K == k:
                training_set.drop(dataset.index[i], inplace = True)
            else:
                test_set.drop(dataset.index[i], inplace = True)
            i += 1
        
        print("Number of training records: %d" % len(training_set))
        print("Number of test records: %d" % len(test_set))

        trainingSet = [list(training_set)[:-1]]
        trainingClasses = [list(training_set)[-1]]

        for index, row in training_set.iterrows():
            trainingSet.append(list(row)[:-1])
            trainingClasses.append(list(row)[-1])
        buildTreeC45.trainTree(trainingSet, trainingClasses,"DecisionTree.xml")
        
        answer = []
        testingSet = [list(test_set)[:-1]]
        for index, row in test_set.iterrows():
            testingSet.append(list(row)[:-1])
            answer.append(list(row)[-1])
        
        prediction = buildTreeC45.testTree("DecisionTree.xml", testingSet)
        error = 0
        for i in range(len(answer)):
            if not answer[i] == prediction[i]:
                error += 1
        acc = 100 - round(float(error) / len(prediction) * 100 , 2)
        
        print("accuracy: %.4f" % acc)

        accuracy.append(acc)
        
    mean_accuracy = math.fsum(accuracy)/10
    print("Accuracy  %f " % (mean_accuracy))
    print("Took %f secs" % (time.clock() - start))

#function to call various methods and generate decision tree
def generate_decision_tree():
    if len(sys.argv) < 3:
        print("Please provide all the arguments.")
    else:
        dataFileName = str(sys.argv[1])
        namesFileName = str(sys.argv[2])
        trainData, testData = read_data(dataFileName, namesFileName)

        trainingSet = [list(trainData)[:-1]]
        trainingClasses = [list(trainData)[-1]]

        for index, row in trainData.iterrows():
            trainingSet.append(list(row)[:-1])
            trainingClasses.append(list(row)[-1])
        
        buildTreeC45.trainTree(trainingSet, trainingClasses,"DecisionTree.xml", True)

        answer = []
        testingSet = [list(testData)[:-1]]
        for index, row in testData.iterrows():
            testingSet.append(list(row)[:-1])
            answer.append(list(row)[-1])
        
        prediction = buildTreeC45.testTree("DecisionTree.xml", testingSet)
        error = 0
        for i in range(len(answer)):
            if not answer[i] == prediction[i]:
                error += 1
        print("Accuracy = ", 100 - round(float(error) / len(prediction) * 100 , 2), "%")

        print('K-Fold cross Validation')
        namesFile = open(namesFileName, "r")
        columns = namesFile.read().split(",")
        dataSet = pd.read_csv(dataFileName, names = columns)
        kFoldValidation(dataSet)
        
        if checkPackage("pydot"):
            xmlData = open('DecisionTree.xml','r').read()
            xmlDict = xmlToDict(ET.fromstring(xmlData))
            graph = plotTree(xmlDict)
            
            graph.write_png('DecisionTree.png')
            if checkPackage("matplotlib"):
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg

                img = mpimg.imread('DecisionTree.png')
                plt.imshow(img)
                plt.show()


# main call to decision tree modules
if __name__=="__main__":
    generate_decision_tree()
    
