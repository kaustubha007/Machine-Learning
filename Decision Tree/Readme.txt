Name: Kaustubh Agnihotri
Student ID: 1001554290

Language used: Python 3.6

Overview:
The program implements C4.5 algorithm to build decision tree and displays it in XML as well as graphical format.
The displaying of graph requires pydot and graphviz packages and hence might not run on all the systems. Check for 
the same has been made in the program. The program also implements k fold cross validation on the dataset provided.

Instructions to run the program:
Enter the following command in command prompt:
python decisionTreeC45.py data_file_name column_file_name

Eg.:
python decisionTreeC45.py house_votes_84.data house_votes_84.columns

Note: Test files have been included along with the image of the generated tree. The program always considers last 
column as class so please ensure to check this in the input files.

Major functions used:
decisionTreeC45.py
Function Name: generate_decision_tree()
Description: Main function which is used to call all the other subroutines to implement various tasks.

Function Name: read_data(dataFileName, namesFileName)
Description: Reads the data and the headers and generates dataframes using pandas. While generating the dataframes the
data is split into two parts i.e. training data and test data using random.rand from numpy ensuring that
the data is split fairly and randomly.

Function Name: kFoldValidation(dataset)
Description: Performs k fold cross validation (10 folds) where the data is split using modulus logic and for every fold
it is ensured that the training and test data are different and the test data is from the particular fold
in progress. Accuracy is calculated for every fold and in the end mean accuracy and time taken for k fold
validation are also displayed.

buildTreeC45.py
Function Name: trainTree(trainDataSet, trainClass, xmlFileName, displayTree = False)
Description: Uses the training data to train and build tree. The tree is built and stored as xml file. Also the tree is 
displayed in xml format too as graph might not be supported on all systems.
			 
Function Name: testTree(xmlFileName, testDataSet)
Description: Uses the trained tree and test data to test and predict the class of the samples. It generates a list of
predicted values for all the samples with the help of some developed functions and returns the list which
is then used to calculate the accuracy of the tree.

Function Name: buildTree(dataSet, classes, parent, attrNames)
Description: Takes input as all the classes and attributes for the training data and generates tree with the help of 
information gain and entropy of the dataset provided. The tree is built using ElementTree from xml.etree
			 
xmlToDict.py
Function Name: xmlToDict(element_tree)
Description: Converts the xmlTree into dictionary format which is later used to plot the tree in a graphical format.

plotTree.py
Function Name: walkXmlDict(graph, xmlDict, parentNode = None)
Description: Used to recursively iterate over the generated dictionary of xml tree and plot is using pydot which
uses graphviz in order to store the graph as image.
			 
