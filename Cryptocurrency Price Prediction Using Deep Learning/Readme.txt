Name: Kaustubh Agnihotri
Student ID: 1001554290

Language used: Python 3.6

Overview:
The program implements machine learning algorithms to predict the cryptocurrency prices using the cryptocurrency financial data.
There are 3 methods implemented viz.:
1) LSTM building blocks for RNN
2) MLP
3) Random Walk
A comparison is done amongst these methods and the results clearly show that the LSTM network performs better than the other two. The program also plots data at every stage so as to make sure that the model is progressing in a right direction. The graphical visualization enables us to see how the data trend flows and how the model is able to predict the prices with a high accuracy. This is a test project and should not be used for making any investments. If used, the user will be completely responsible for the outcomes.

Instructions to run the program:
Enter the following command in command prompt:
python Cryptocurrency_Price_Prediction.py

Note: Test graphs and models for ethereum have been included along with the code. The project currently works for 5 currencies and can be extended with certain minor code changes.

Major functions used:
Cryptocurrency_Price_Prediction.py
Function Name: predictCurrencyPrices()
Description: Main function which is used to allow user to select a currency and then call all the other subroutines to implement various tasks. This function also calculates the date upon which the training and testing data is to be splitted.

Function Name: fetchData(currURL)
Description: Fetches live rates using the url from coinmarketcap.com and cleans the data by handling all the data inconsistencies.

Manipulate_Data.py
Function Name: plotData(currName, currData, splitDate)
Description: Plots the entire data for the closing price vs date and the volume traded. This allows us to visualize the shift in the data values.
			 
Function Name: plotTrainTestData(currName, currData, currImage, splitDate)
Description: Splits the data into two sections of training and testing and plots these sections on a graph.

Function Name: plotNormalDist(currName, currData, currImage, splitDate)
Description: This function makes sure that the data is normally distributed before we move ahead with further data manipulations and model generation.
			 
Few more functions are used to plot the results as we progress.
			 
Build_LSTM.py
Function Name: buildLSTMData(currName, currData, splitDate, daysPrediction = 5)
Description: We first extract a few features from the raw data which we further use for building our model. Then the function formulates the data in a way such that we can use it for the training of our model. The trained model is used on the testing set to predict the prices. We plot the predictions and calculate the	accuracy. The function is then used to predict the prices for 5 days and see the trend. Then we use 25 different random seeds to build and train 25 different LSTM, MLP and Random Walk models each. Then we compare the prices predicted by these models and display the obtained results.

Function Name: buildLSTMModel(inputs, output_size, neurons, activ_func = "linear", dropout = 0.25, loss = "mae", optimizer = "adam", rand_seed = 202, batch_size = 1)
Description: This function takes in various inputs to build a RNN with LSTM building blocks. We have used the default parameters as shown above to build the model. This model is then used for training and predicting prices.
			 
Function Name: buildMLPModel(inputs, output_size, neurons, activ_func = "relu", dropout = 0.25, loss = "mae", optimizer = "adam", rand_seed = 202, batch_size = 1)
Description: This function takes in various inputs to build a neural network with the default parameters shown above.

Function Name: predictTestPrice(currName, modelData, dataSet, windowSize, splitDate, LSTM_inputs, LSTM_model)
Description: This function is used to predict the prices for the cryptocurrency using the trained model and then returns these prices which we plot in the graph.
