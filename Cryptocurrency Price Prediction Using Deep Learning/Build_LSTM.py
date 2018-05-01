import numpy as np
import datetime
import h5py
import os
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Activation, LSTM, Dense, Dropout, Flatten
from pandas import ExcelWriter
from Manipulate_Data import plotError, plotPredictedPrices, plotLSTMandMLPandRndWalkMAE


def buildLSTMModel(inputs, output_size, neurons, activ_func = "linear", dropout = 0.25, loss = "mae", optimizer = "adam", rand_seed = 202, batch_size = 1):
    # random seed for reproducibility
    np.random.seed(rand_seed)
    #create LSTM model
    model = Sequential()

    model.add(LSTM(neurons, return_sequences = True, batch_input_shape = (batch_size, inputs.shape[1], inputs.shape[2]), stateful=True))
    model.add(LSTM(neurons, return_sequences = True, stateful=True))
    model.add(LSTM(neurons, stateful=True))
    model.add(Dropout(dropout))
    model.add(Dense(units = output_size))
    model.add(Activation(activ_func))

    model.compile(loss = loss, optimizer = optimizer)

    return model

def buildMLPModel(inputs, output_size, neurons, activ_func = "relu", dropout = 0.25, loss = "mae", optimizer = "adam", rand_seed = 202, batch_size = 1):
    np.random.seed(rand_seed)
    # create Multilayer Perceptron model
    model = Sequential()
    model.add(Dense(neurons, input_shape = (inputs.shape[1], inputs.shape[2])))
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(units = 1))
    model.add(Activation(activ_func))
    
    model.compile(loss = loss, optimizer = optimizer)

    return model


def getInputs(dataSet, windowSize, acceptCols, currName, flagPredictFuture = False, LSTM_Data_Inputs = None, daysPrediction = 5):
    if not flagPredictFuture:
        LSTM_Data_Inputs = []
        for i in range(len(dataSet) - windowSize):
            tempSet = dataSet[i : (i + windowSize)].copy()
            for col in acceptCols:
                tempSet.loc[:, col] = tempSet[col] / tempSet[col].iloc[0] - 1
            LSTM_Data_Inputs.append(tempSet)
        LSTM_Data_Outputs = (dataSet[currName + '_Close'][windowSize:].values / dataSet[currName + '_Close'][:-windowSize].values) - 1
        LSTM_Data_Inputs = [np.array(LSTM_Data_Input) for LSTM_Data_Input in LSTM_Data_Inputs]
        LSTM_Data_Inputs = np.array(LSTM_Data_Inputs)
        LSTM_Data_Inputs = np.nan_to_num(LSTM_Data_Inputs)
    else:
        LSTM_Data_Outputs = []
        for i in range(windowSize, len(dataSet[currName + '_Close']) - daysPrediction):
            LSTM_Data_Outputs.append((dataSet[currName + '_Close'][i : i + daysPrediction].values/
                                          dataSet[currName + '_Close'].values[i - windowSize]) - 1)
    
    return LSTM_Data_Inputs, LSTM_Data_Outputs

def predictTrainingPrice(currName, modelData, dataSet, windowSize, splitDate, LSTM_inputs, LSTM_model):
    plotVars = []
    plotVars.append(modelData[modelData['Date'] < splitDate]['Date'][windowSize:].astype(datetime.datetime))
    plotVars.append(dataSet[currName + '_Close'][windowSize:])
    plotVars.append(modelData[modelData['Date']< splitDate]['Date'][windowSize:].astype(datetime.datetime))
    plotVars.append(((np.transpose(LSTM_model.predict(LSTM_inputs, batch_size = 1))+1) * dataSet[currName + '_Close'].values[:-windowSize])[0])
    plotVars.append('MAE: %.4f' % np.mean(np.abs((np.transpose(LSTM_model.predict(LSTM_inputs, batch_size = 1)) + 1) - \
                (dataSet[currName + '_Close'].values[windowSize:])/(dataSet[currName + '_Close'].values[:-windowSize]))))
    plotVars.append(modelData[modelData['Date']< splitDate]['Date'][windowSize:].astype(datetime.datetime))
    plotVars.append(dataSet[currName + '_Close'][windowSize:])
    plotVars.append(modelData[modelData['Date']< splitDate]['Date'][windowSize:].astype(datetime.datetime))
    plotVars.append(((np.transpose(LSTM_model.predict(LSTM_inputs, batch_size = 1))+1) * dataSet[currName + '_Close'].values[:-windowSize])[0])

    return plotVars

def predictTestPrice(currName, modelData, dataSet, windowSize, splitDate, LSTM_inputs, LSTM_model):
    plotVars = []
    plotVars.append(modelData[modelData['Date']>= splitDate]['Date'][windowSize:].astype(datetime.datetime))
    plotVars.append(dataSet[currName + '_Close'][windowSize:])
    plotVars.append(modelData[modelData['Date'] >= splitDate]['Date'][windowSize:].astype(datetime.datetime))
    plotVars.append(((np.transpose(LSTM_model.predict(LSTM_inputs, batch_size = 1))+1) * dataSet[currName + '_Close'].values[:-windowSize])[0])
    plotVars.append('MAE: %.4f'%np.mean(np.abs((np.transpose(LSTM_model.predict(LSTM_inputs, batch_size = 1))+1) - \
                (dataSet[currName + '_Close'].values[windowSize:])/(dataSet[currName + '_Close'].values[:-windowSize]))))

    return plotVars

def predictFuturePrices(currName, modelData, dataSet, windowSize, splitDate, LSTM_inputs, LSTM_model, daysPrediction):
    plotVars = []
    predictedPrices = ((LSTM_model.predict(LSTM_inputs, batch_size = 1)[:-daysPrediction][::daysPrediction]+1)*\
                   dataSet[currName + '_Close'].values[:-(windowSize + daysPrediction)][::5].reshape(int(np.ceil((len(LSTM_inputs) - daysPrediction)/float(daysPrediction))), 1))
    plotVars.append(modelData[modelData['Date'] >= splitDate]['Date'][windowSize:].astype(datetime.datetime))
    plotVars.append(dataSet[currName + '_Close'][windowSize:])
    plotPredictionVars = []
    for i in range(len(predictedPrices)):
        plotPredictionVars.append([modelData[modelData['Date'] >= splitDate]['Date'][windowSize:].astype(datetime.datetime)[i * daysPrediction : i * daysPrediction + daysPrediction], predictedPrices[i]])
    plotVars.append(plotPredictionVars)
    
    return plotVars

# A function to extract data features and call other functions to build, train and predict prices
def buildLSTMData(currName, currData, splitDate, daysPrediction = 5):
    kwargs = {currName + '_' + 'close_off_high': lambda x: 2 * (x[currName + '_' + 'High'] - x[currName + '_' + 'Close']) / (x[currName + '_' + 'High'] \
                    - x[currName + '_' + 'Low']) - 1, currName + '_' + 'volatility': lambda x: (x[currName + '_' + 'High'] - x[currName + '_' + 'Low']) \
                    / (x[currName + '_' + 'Open'])}
    currData = currData.assign(**kwargs)
    modelData = currData[['Date'] + [currName + '_' + feature for feature in ['Close', 'Volume', 'close_off_high', 'volatility']]]
    # Need to reverse the data frame so that subsequent rows represent later timepoints
    modelData = modelData.sort_values(by = 'Date')
    print("A look at the data with features")
    print(modelData.head())
    # We don't need the date columns anymore
    trainingData, testData = modelData[modelData['Date'] < splitDate], modelData[modelData['Date'] >= splitDate]
    trainingData = trainingData.drop('Date', 1)
    testData = testData.drop('Date', 1)
    # Setting the window size for the LSTM blocks
    windowSize = 10
    acceptCols = [currName + '_' + metric for metric in ['Close', 'Volume']]
    LSTM_TrainingInputs, LSTM_TrainingOutputs = getInputs(trainingData, windowSize, acceptCols, currName)
    LSTM_TestInputs, LSTM_TestOutputs = getInputs(testData, windowSize, acceptCols, currName)
    
    
    # Initialise model architecture
    LSTM_Model = buildLSTMModel(LSTM_TrainingInputs, output_size=1, neurons = 20)
    # Model output is next price normalised to 10th previous closing price
    LSTM_TrainingOutputs = (trainingData[currName + '_' + 'Close'][windowSize:].values / trainingData[currName + '_' + 'Close'][:-windowSize].values) - 1
    
    # Train model on data
    # LSTM_History contains information on the training error per epoch
    print('Training model with the training data.')
    LSTM_History = LSTM_Model.fit(LSTM_TrainingInputs, LSTM_TrainingOutputs, validation_split = 0.33, epochs = 50, batch_size = 1, verbose = 2, shuffle = True)
    plotError(currName, LSTM_History, LSTM_Model)
    plotTrainVars = predictTrainingPrice(currName, modelData, trainingData, windowSize, splitDate, LSTM_TrainingInputs, LSTM_Model)
    plotPredictedPrices(currName, plotTrainVars, 'Training', trainFlag = True, flagPredictFuture = False)
    plotTestVars = predictTestPrice(currName, modelData, testData, windowSize, splitDate, LSTM_TestInputs, LSTM_Model)
    plotPredictedPrices(currName, plotTestVars, 'Test', trainFlag = False, flagPredictFuture = False)
##    print("\nOverall Accuracy = " + str((1 - (float(plotTestVars[4].split(' ')[1]))) * 100))
    overallAccuracy = (1 - np.sum(np.absolute((plotTestVars[3] - plotTestVars[1])) / plotTestVars[1]) / len(plotTestVars[1])) * 100
    print("\nOverall Accuracy = " + str(overallAccuracy))
    
    print("\nNow predicting prices for the next 5 days.")
    LSTM_Model = buildLSTMModel(LSTM_TrainingInputs, output_size = daysPrediction, neurons = 20)
    # Model output is next 5 prices normalised to 10th previous closing price
    LSTM_TrainingInputs, LSTM_TrainingOutputs = getInputs(trainingData, windowSize, acceptCols, currName, True, LSTM_TrainingInputs, daysPrediction)
    LSTM_TrainingOutputs = np.array(LSTM_TrainingOutputs)
    LSTM_History = LSTM_Model.fit(LSTM_TrainingInputs[:-daysPrediction], LSTM_TrainingOutputs, validation_split = 0.33, epochs = 50, batch_size = 1, verbose = 2, shuffle = True)
    plotFutureVars = predictFuturePrices(currName, modelData, testData, windowSize, splitDate, LSTM_TestInputs, LSTM_Model, daysPrediction)
    plotPredictedPrices(currName, plotFutureVars, 'Test', trainFlag = False, flagPredictFuture = True, daysPrediction = daysPrediction)

    print("Now training models for 25 different random seeds. This might take a while...")
    print("LSTM Models")
    directories = ['Models/' + currName + '/LSTM', 'Models/' + currName + '/MLP']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    # Run this code once to train 25 LSTM models and then predict prices for each to compare with MLP
    # The trained models are saved so that we need not train them again
    # Currently models have been trained and saved for ethereum
    
##    for iRandSeed in range(237, 262):
##        print("Training model with random seed: " + str(iRandSeed))
##        tempLSTM_Model = buildLSTMModel(LSTM_TrainingInputs, output_size = 1, neurons = 20, rand_seed = iRandSeed)
##        LSTM_TrainingOutputs = (trainingData[currName + '_' + 'Close'][windowSize:].values / trainingData[currName + '_Close'][:-windowSize].values) - 1
##        tempLSTM_Model.fit(LSTM_TrainingInputs, LSTM_TrainingOutputs, epochs = 50, batch_size = 1, verbose = 2, shuffle = True)
##        tempLSTM_Model.save(directories[0] + '/LSTM_model_randseed_%d.h5' % iRandSeed)
    
    LSTM_predictions = []
    for iRandSeed in range(237, 262):
        print("Predicting using model trained with random seed: " + str(iRandSeed))
        np.random.seed(iRandSeed)
        tempLSTM_Model = load_model(directories[0] + '/LSTM_model_randseed_%d.h5' % iRandSeed)
        LSTM_predictions.append(np.mean(abs(np.transpose(tempLSTM_Model.predict(LSTM_TestInputs, batch_size = 1))-
                (testData[currName + '_Close'].values[windowSize:]/testData[currName + '_Close'].values[:-windowSize]-1))))

    print("LSTM predictions completed")

    print("MLP Models")
    # Run this code once to train 25 LSTM models and then predict prices for each to compare with MLP
    #The trained models are saved so that we need not train them again
    #Currently models have been trained and saved for ethereum
    
##    for iRandSeed in range(237, 262):
##        print("Training model with random seed: " + str(iRandSeed))
##        tempMPL_Model = buildMLPModel(LSTM_TrainingInputs, output_size = 1, neurons = 20, rand_seed = iRandSeed)
##        LSTM_TrainingOutputs = (trainingData[currName + '_' + 'Close'][windowSize:].values / trainingData[currName + '_' + 'Close'][:-windowSize].values) - 1
##        tempMPL_Model.fit(LSTM_TrainingInputs, LSTM_TrainingOutputs, epochs=50, batch_size=1, verbose=2, shuffle = True)
##        tempMPL_Model.save(directories[1] + '/MLP_model_randseed_%d.h5' % iRandSeed)

    MLP_predictions = []
    for iRandSeed in range(237, 262):
        print("Predicting using model trained with random seed: " + str(iRandSeed))
        np.random.seed(iRandSeed)
        tempLSTM_Model = load_model(directories[1] + '/MLP_model_randseed_%d.h5' % iRandSeed)
        MLP_predictions.append(np.mean(abs(np.transpose(tempLSTM_Model.predict(LSTM_TestInputs))-
                (testData[currName + '_Close'].values[windowSize:]/testData[currName + '_Close'].values[:-windowSize]-1))))

    print("MLP predictions completed")
    print("Generating random walk predictions")
    RandomWalk_predictions = []
    for iRandSeed in range(237, 262):
        np.random.seed(iRandSeed)
        ndWalkMean, rndWalkSD = np.mean(currData[currData['Date'] < splitDate][currName + '_day_diff'].values), \
                             np.std(currData[currData['Date'] < splitDate][currName + '_day_diff'].values)
        RandomWalk_predictions.append(np.mean(np.abs((np.random.normal(ndWalkMean, rndWalkSD, len(testData) - windowSize) + 1)-
                           np.array(testData[currName + '_Close'][windowSize:]) / np.array(testData[currName + '_Close'][:-windowSize]))))

    print("Random walk predictions completed")
    plotLSTMandMLPandRndWalkMAE(currName, LSTM_predictions, MLP_predictions, RandomWalk_predictions)

    rangeLSTM_MAE = max(LSTM_predictions) - min(LSTM_predictions)
    rangeMLP_MAE = max(MLP_predictions) - min(MLP_predictions)
    rangeRndWlk_MAE = max(RandomWalk_predictions) - min(RandomWalk_predictions)
    
    sumLSTM_MAE = 0
    sumMLP_MAE = 0
    sumRndWlk_MAE = 0
    for errLSTM, errMLP, errRndWlk in zip(LSTM_predictions, MLP_predictions, RandomWalk_predictions):
        sumLSTM_MAE += errLSTM
        sumMLP_MAE += errMLP
        sumRndWlk_MAE += errRndWlk

    avgErrorLSTM = sumLSTM_MAE / len(LSTM_predictions)
    avgErrorMLP = sumMLP_MAE / len(MLP_predictions)
    avgErrorRndWlk = sumRndWlk_MAE / len(RandomWalk_predictions)

    print("\nResult:")
    print("Accuracy: " + str(overallAccuracy))
    print("Mean Absolute Error: " + str(plotTestVars[4].split(' ')[1]))
    print("Avg/Range MAE for 3 methods: ")
    print("1.1 LSTM Avg MAE:\t\t\t" + str(avgErrorLSTM))
    print("1.2 LSTM MAE Range:\t\t\t" + str(rangeLSTM_MAE))
    print("2.1 MLP Avg MAE:\t\t\t" + str(avgErrorMLP))
    print("2.2 MLP MAE Range:\t\t\t" + str(rangeMLP_MAE))
    print("3.1 Random Walk Avg MAE:\t\t" + str(avgErrorRndWlk))
    print("3.2 Random Walk MAE Range:\t\t" + str(rangeRndWlk_MAE))
    print("Completed.")
    
    
