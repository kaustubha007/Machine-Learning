from PIL import Image
import matplotlib.pyplot as plt
import datetime
import os
import numpy as np

def testDataRandomWalk(currName, currData, currImage, splitDate, rndSteps):
    np.random.seed(202)
    rndWalk = []
    for step, step1 in enumerate(rndSteps):
        if step == 0:
            rndWalk.append(currData[currData['Date']< splitDate][currName + '_Close'].values[0] * (step1 + 1))
        else:
            rndWalk.append(rndWalk[step - 1] * (step1 + 1))

    fig, (ax1) = plt.subplots()
    ax1.set_xticks([datetime.date(2017, i + 1, 1) for i in range(12)])
    ax1.set_xticklabels('')
    ax1.plot(currData[currData['Date'] >= splitDate]['Date'].astype(datetime.datetime),
             currData[currData['Date'] >= splitDate][currName + '_Close'].values, label = 'Actual')
    ax1.plot(currData[currData['Date'] >= splitDate]['Date'].astype(datetime.datetime),
             rndWalk[::-1], label = 'Predicted')
    
    ax1.set_title('Full Interval Random Walk')
    ax1.set_ylabel('Price ($)', fontsize = 12)
    ax1.legend(bbox_to_anchor = (0.1, 1), loc = 2, borderaxespad = 0., prop = {'size': 14})
    fig.figimage(currImage.resize((int(currImage.size[0]*0.65), int(currImage.size[1]*0.65)), Image.ANTIALIAS), 200, 260, zorder = 3,alpha = .5)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.savefig(directory + '/05 ' + currName.upper() + ' Full Interval Random Walk.png', bbox_inches = 'tight')
    plt.show()
    


def singlePointRandomWalk(currName, currData, currImage, splitDate):
    np.random.seed(202)
    rndWalkMean, rndWalkSD = np.mean(currData[currData['Date'] < splitDate][currName + '_day_diff'].values), \
                             np.std(currData[currData['Date'] < splitDate][currName + '_day_diff'].values)
    rndSteps = np.random.normal(rndWalkMean, rndWalkSD, 
                    (max(currData['Date']).to_pydatetime() - datetime.datetime.strptime(splitDate, '%Y-%m-%d')).days + 1)

    fig, (ax1) = plt.subplots()
    ax1.set_xticks([datetime.date(2017, i + 1, 1) for i in range(12)])
    ax1.set_xticklabels('')
    ax1.plot(currData[currData['Date'] >= splitDate]['Date'].astype(datetime.datetime),
         currData[currData['Date'] >= splitDate][currName + '_Close'].values, label = 'Actual')
    ax1.plot(currData[currData['Date'] >= splitDate]['Date'].astype(datetime.datetime),
          currData[(currData['Date']+ datetime.timedelta(days=1)) >= splitDate][currName + '_Close'].values[1:] * 
         (1+rndSteps), label = 'Predicted')
    ax1.set_title('Single Point Random Walk (Test Set)')
    ax1.set_ylabel('Price ($)', fontsize = 12)
    ax1.legend(bbox_to_anchor = (0.1, 1), loc = 2, borderaxespad = 0., prop = {'size': 14})
    
    fig.figimage(currImage.resize((int(currImage.size[0]*0.65), int(currImage.size[1]*0.65)), Image.ANTIALIAS), 200, 260, zorder=3,alpha=.5)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.savefig(directory + '/04 ' + currName.upper() + ' Single Point Random Walk.png', bbox_inches = 'tight')
    plt.show()
    return rndSteps


def plotNormalDist(currName, currData, currImage, splitDate):
    fig, (ax1) = plt.subplots()
    ax1.hist(currData[currData['Date']< splitDate][currName + '_day_diff'].values, bins=100)
    ax1.set_title('Daily Price Changes')
    fig.figimage(currImage.resize((int(currImage.size[0]*0.65), int(currImage.size[1]*0.65)), Image.ANTIALIAS), 200, 260, zorder=3,alpha=.5)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.savefig(directory + '/03 ' + currName.upper() + ' Normal Distribution.png', bbox_inches = 'tight')
    plt.show()

def plotTrainTestData(currName, currData, currImage, splitDate):
    fig, (ax1) = plt.subplots()
    ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
    ax1.set_xticklabels('')
    ax1.plot(currData[currData['Date'] < splitDate]['Date'].astype(datetime.datetime),
             currData[currData['Date'] < splitDate][currName + '_Close'], 
             color='#32CD32', label='Training')
    ax1.plot(currData[currData['Date'] >= splitDate]['Date'].astype(datetime.datetime),
             currData[currData['Date'] >= splitDate][currName + '_Close'], 
             color='#0033FF', label='Test')
    ax1.set_xticklabels('')
    ax1.set_ylabel('Closing Price ($)', fontsize = 12)
    plt.tight_layout()
    ax1.legend(bbox_to_anchor=(0.03, 1), loc=2, borderaxespad  = 0., prop = {'size': 14})
    fig.figimage(currImage.resize((int(currImage.size[0]*0.65), int(currImage.size[1]*0.65)), Image.ANTIALIAS), 
                 200, 260, zorder = 3, alpha = .5)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.savefig(directory + '/02 ' + currName.upper() + ' Train and Test Division.png', bbox_inches = 'tight')
    plt.show()

def plotData(currName, currData, splitDate):
    global directory
    directory = 'TestGraphs/'
    directory += currName
    if not os.path.exists(directory):
        os.makedirs(directory)
    imgFile = "img/" + currName + ".png"
    currImage = Image.open(imgFile)
    currData.columns =[currData.columns[0]]+[currName + '_' + i for i in currData.columns[1:]]
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3, 1]})
    ax1.set_ylabel('Closing Price ($)', fontsize=12)
    ax2.set_ylabel('Volume ($ bn)', fontsize=12)
    ax2.set_yticks([int('%d000000000'%i) for i in range(10)])
    ax2.set_yticklabels(range(10))
    ax1.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
    ax1.set_xticklabels('')
    ax2.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
    ax2.set_xticklabels([datetime.date(i, j, 1).strftime('%b %Y')  for i in range(2013, 2019) for j in [1, 7]])
    ax1.plot(currData['Date'].astype(datetime.datetime),currData[currName + '_Open'])
    ax2.bar(currData['Date'].astype(datetime.datetime).values, currData[currName + '_Volume'].values)
    fig.tight_layout()
    fig.figimage(currImage.resize((int(currImage.size[0]*0.65), int(currImage.size[1]*0.65)), Image.ANTIALIAS), 100, 120, zorder=3,alpha=.5)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.savefig(directory + '/01 ' + currName.upper() + ' Data.png', bbox_inches = 'tight')
    plt.show()
    plotTrainTestData(currName, currData, currImage, splitDate)
    kwargs = {currName + '_' + 'day_diff': lambda x: (x[currName + '_' + 'Close'] - x[currName + '_' + 'Open']) / x[currName + '_' + 'Open']}
    currData = currData.assign(**kwargs)
    plotNormalDist(currName, currData, currImage, splitDate)
    rndSteps = singlePointRandomWalk(currName, currData, currImage, splitDate)
    testDataRandomWalk(currName, currData, currImage, splitDate, rndSteps)
    return currData
    
def plotError(currName, historyData, model):
    imgFile = "img/" + currName + ".png"
    currImage = Image.open(imgFile)
    fig, (ax1) = plt.subplots()

    ax1.plot(historyData.epoch, historyData.history['loss'])
    ax1.set_title('Training Error')

    if model.loss == 'mae':
        ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize = 12)
    # just in case you decided to change the model loss calculation
    else:
        ax1.set_ylabel('Model Loss', fontsize=12)
    ax1.set_xlabel('# Epochs', fontsize=12)
    fig.figimage(currImage.resize((int(currImage.size[0]*0.65), int(currImage.size[1]*0.65)), Image.ANTIALIAS), 
                 200, 260, zorder = 3, alpha = .5)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.savefig(directory + '/06 ' + currName.upper() + ' Mean Absolute Error.png', bbox_inches = 'tight')
    plt.show()


def plotLSTMandMLPandRndWalkMAE(currName, LSTM_predictions, MLP_predictions, RandomWalk_predictions):
    imgFile = "img/" + currName + ".png"
    currImage = Image.open(imgFile)
    fig, (ax1) = plt.subplots()
    ax1.boxplot([LSTM_predictions, MLP_predictions, RandomWalk_predictions], widths=0.75)
    ax1.set_ylim([0, 0.4])
    ax1.set_xticklabels(['LSTM', 'MLP', 'Random Walk'])
    ax1.set_title(currName + ' Test Set (25 runs)')
    ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    fig.figimage(currImage.resize((int(currImage.size[0]*0.65), int(currImage.size[1]*0.65)), Image.ANTIALIAS), 
                         200, 260, zorder = 3, alpha = .5)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.savefig(directory + '/10 ' + currName.upper() + ' MAE Comparison for 25 runs with 3 methods.png', bbox_inches = 'tight')
    plt.show()


def plotPredictedPrices(currName, plotVars, setName, trainFlag, flagPredictFuture = False, daysPrediction = 5):
    imgFile = "img/" + currName + ".png"
    currImage = Image.open(imgFile)
    if trainFlag:
        fig, (ax1) = plt.subplots()
        ax1.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 5, 9]])
        ax1.set_xticklabels([datetime.date(i, j, 1).strftime('%b %Y')  for i in range(2013, 2019) for j in [1, 5, 9]])
        
        ax1.plot(plotVars[0], plotVars[1], label = 'Actual')
        ax1.plot(plotVars[2], plotVars[3], label = 'Predicted')
        ax1.set_title(setName + ' Set: Single Timepoint Prediction')
        ax1.set_ylabel('Closing Price ($)', fontsize = 12)
        ax1.legend(bbox_to_anchor = (0.15, 1), loc = 2, borderaxespad = 0., prop = {'size': 14})
        ax1.annotate(plotVars[4], xy = (0.75, 0.9), xycoords = 'axes fraction', xytext = (0.75, 0.9), textcoords = 'axes fraction')
        fig.figimage(currImage.resize((int(currImage.size[0]*0.65), int(currImage.size[1]*0.65)), Image.ANTIALIAS), 
                     200, 260, zorder = 3, alpha = .5)
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.savefig(directory + '/07 ' + currName.upper() + ' Training Set Prediction LSTM.png', bbox_inches = 'tight')
        plt.show()
    else:
        if flagPredictFuture:
            predictionColors = ["#FF69B4", "#5D6D7E", "#F4D03F","#A569BD","#45B39D"]
            fig, ax1 = plt.subplots()
            ax1.set_xticks([datetime.date(i, j, 1) for i in range(2017, 2019) for j in [1, 3, 6, 9, 12]])
            ax1.set_xticklabels([datetime.date(i, j, 1).strftime('%b %Y')  for i in range(2017, 2019) for j in [1, 3, 6, 9, 12]])
            ax1.plot(plotVars[0], plotVars[1], label='Actual')
            for i in range(len(plotVars[2])):
                if i < 5:
                    ax1.plot(plotVars[2][i][0], plotVars[2][i][1], color = predictionColors[i % 5], label = "Predicted")
                else:
                    ax1.plot(plotVars[2][i][0], plotVars[2][i][1], color = predictionColors[i % 5])
                    
            ax1.set_title(setName + ' Set: Single Timepoint Prediction', fontsize = 13)
            ax1.set_ylabel('Closing Price ($)', fontsize = 12)
            ax1.legend(bbox_to_anchor = (0.1, 1), loc = 2, borderaxespad = 0., prop = {'size': 14})
            fig.figimage(currImage.resize((int(currImage.size[0]*0.65), int(currImage.size[1]*0.65)), Image.ANTIALIAS), 
                         200, 260, zorder = 3, alpha = .5)
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
            plt.savefig(directory + '/09 ' + currName.upper() + ' Test Set 5 Predictions LSTM.png', bbox_inches = 'tight')
            plt.show()
        else:
            fig, ax1 = plt.subplots()
            ax1.set_xticks([datetime.date(i, j, 1) for i in range(2017, 2019) for j in [1, 3, 6, 9, 12]])
            ax1.set_xticklabels([datetime.date(i, j, 1).strftime('%b %Y')  for i in range(2017, 2019) for j in [1, 3, 6, 9, 12]])
            ax1.plot(plotVars[0], plotVars[1], label='Actual')
            ax1.plot(plotVars[2], plotVars[3], label='Predicted')
            ax1.annotate(plotVars[4], xy = (0.75, 0.9),  xycoords = 'axes fraction', xytext = (0.75, 0.9), textcoords = 'axes fraction')
            ax1.set_title(setName + ' Set: Single Timepoint Prediction', fontsize = 13)
            ax1.set_ylabel('Closing Price ($)', fontsize = 12)
            ax1.legend(bbox_to_anchor = (0.1, 1), loc = 2, borderaxespad = 0., prop = {'size': 14})
            fig.figimage(currImage.resize((int(currImage.size[0]*0.65), int(currImage.size[1]*0.65)), Image.ANTIALIAS), 
                         200, 260, zorder = 3, alpha = .5)
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
            plt.savefig(directory + '/08 ' + currName.upper() + ' Test Set Prediction LSTM.png', bbox_inches = 'tight')
            plt.show()
