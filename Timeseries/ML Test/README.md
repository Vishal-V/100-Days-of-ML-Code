# Krishihub ML Problem Statement

The given problem statement asks for a time-series prediction of 3 continuous value features. Thus, I have used time series forecasting using ARIMA to predict the prices for the next 30 days.

## Algorithm
I have used an ARIMA (Auto Regressive Integrated Moving Average) model with rolling-forecasting to predict the values.  
ARIMA models are excellent for temporal predictions with seasonal distributions. The data shows seasonal variability and hence I used this model. The validation is a walk-forward forecast with a Root Mean Square cost function.
  
## Logic
My machine learning model is the ARIMA model as the data includes a seasonal temporal shift. The data is non-stationary as it has trends in the distribution and thus we need to integrate the differences as used in Box-Jenkins approach in order to make the data stationary for accuarte predictions. I have used a difference factor of 292 days as the there are on average 292 days for every years' values.
  
ARIMA is :  
`Auto-Regression(AR)` - The values of a given time series data are regressed on their own lagged values, which is indicated by the “p” value in the ARIMA model.  
Differencing `(I-for Integrated)` - This involves differencing the time series data to remove the trend and convert a non-stationary time series to a stationary one.   
`Moving Average (MA)` - The moving average nature of the ARIMA model is represented by the “q” value which is the number of lagged values of the error term.  
Since the azd data requires p,q and r for predictions, my ARIMA model is (1,1,1) which is (p,d,q)  
Therefore ARIMA(1,1,1)  
  
Walk-forward validation is extremely accurate as it provides every iteration with all the available data. This is computationally intensive and hence can used only for small datasets. Since the 'azd' dataset has 3805 observations, I have used walk-forward method of forecasting  
The cost function used to evaluate the model is a Root-Mean-Squared function
    
## Possible Improvements
1. Grid search can be used to improve the ARIMA values for a much better accuracy.  
2. The mean squared error can be added to the inverse-difference values to remove the bias (lagging)  
3. Using an lstm or a memory network for time-series predictions  