import pandas as pd
import time
from Manipulate_Data import plotData
from Build_LSTM import buildLSTMData

# A function to fetch the live data of cryptocurrency prices from coinmarketcap.com
# Also handles the data inconsistencies for certain columns
def fetchData(currURL):
    currData = pd.read_html(currURL)[0]
    # convert the date string to the correct date format
    currData = currData.assign(Date = pd.to_datetime(currData['Date']))
    # when Volume/MarketCap is equal to '-' convert it to 0
    try:
        cntBlanksVol = 0
        cntBlanksVol = currData['Volume'].str.contains('-').sum()
    except Exception as e:
        pass
    
    try:
        cntBlanksMkt = 0
        cntBlanksMkt = currData['Market Cap'].str.contains('-').sum()
    except Exception as e:
        pass
    if cntBlanksVol > 0:
        currData.loc[currData['Volume'] == "-", 'Volume'] = 0
        print("Data inconsistencies handled.")
    if cntBlanksMkt > 0:
        currData.loc[currData['Market Cap'] == "-", 'Market Cap'] = 0
        print("Data inconsistencies handled.")

    if cntBlanksVol == 0 and cntBlanksMkt == 0:
        print("No data inconsistencies found.")
    # convert to int
    currData['Volume'] = currData['Volume'].astype('int64')
    currData['Market Cap'] = currData['Market Cap'].astype('int64')

    return currData

def predictCurrencyPrices():
    # A dictionary with currency choices and their urls to access data from the web
    dictCurrencies = {'1': ['Bitcoin', 'bitcoin'],
                              '2': ['Ethereum', 'ethereum'],
                              '3': ['Ripple', 'ripple'],
                              '4': ['Bitcoin Cash', 'bitcoin-cash'],
                              '5': ['Litecoin', 'litecoin']}
    while True:
        for num, curr in dictCurrencies.items():
            print(num + " " + curr[0])
        selCurr = input("Enter the currency number for which you want to predict the price: ")
        if int(selCurr) > 0 and int(selCurr) < 6:
            print("\nPredicting price for " + dictCurrencies[selCurr][0])
            currName = dictCurrencies[selCurr][1]
            # Build url to fetch live market data
            currURL = "https://coinmarketcap.com/currencies/" + currName + "/historical-data/?start=20130428&end="+time.strftime("%Y%m%d")
            print("Fetching data from coinmarketcap.com")
            print("Loading data.....")
            currData = fetchData(currURL)
            currData = currData[currData.Volume != 0]
            # Split the data for training and testing purposes
            splitIndex = int((len(currData) * 20) / 100)
            splitDate = str(currData.iloc[splitIndex]['Date'].date())
            # Function call to plot and visualize various data attributes
            currData = plotData(currName, currData, splitDate)

            # Build and train model and use it for the predictions
            buildLSTMData(currName, currData, splitDate)
            
        else:
            print("Please enter valid choice.")
# Main function to take currency choice as input and then further call various functions to process and get the output
if __name__ == "__main__":
    predictCurrencyPrices()
