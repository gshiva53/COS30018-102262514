<h1><p align="center"> 
    Machine learning 3
</h1>

**Name:** Shiva Gupta
**Student ID:** 102262514 
**Unit Code:** COS30018
**Task Number:** 6
**Repo link:** https://github.com/gshiva53/COS30018-102262514/tree/main/pythonProject

****

This report documents the steps taken to develop an ensemble modelling approach for predicting the stock prices. 

[Source: https://towardsdatascience.com/time-series-forecasting-predicting-stock-prices-using-an-arima-model-2e3b3080bd70]

### Understanding the ARIMA model

ARIMA means AutoRegressive Integrated Moving Average model. It is capable of capturing a suite of different standard temporal structures in time-series data. The parameters for this model are;

```python
p = number of lag observations 
d = degree of differencing 
q = size/width of the moving average window. 
```

During this report we will trial out different hyperparameter configurations for this model and document our findings. 

##### Imports 

The additional dependencies needed for this tasks are as follows. 

```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
```

NOTE: The source article lists the `statsmodels.tsa.arima.model` as `statsmodels.tsa.arima_model` but this was updated recently and the aforementioned import is the correct one. 


****

### Creating the Model 

To create the model we need training and test data. In our project we are predicting the closing stock prices for Facebook. So the data that we are importing from Yahoo finance can be separated in training and testing data and can be further used for the model. 

1. We are going to take 70% of the data for training and 30% for testing and the prediction is only for closing values for now. 

   ```python
   train_data, test_data = data[0:int(len(data)*0.7)], data[int(len(data)*0.7):]
   
   training_data = train_data['Close'].values
   test_data = test_data['Close'].values
   ```

2. Some additional variables that will be needed are: 

   ```python
   model_predictions = []
   history = [x for x in training_data]
   N_test_observations = len(test_data)
   ```

3. `history` is needed to manually keep track of all observations in a list which is then seeded with the training data and new observations are added at each iteration. 

4. Building the model. 

   ```python
   for time_point in range(N_test_observations):
       model = ARIMA(history, order=(4, 1, 0))
       model_fit = model.fit()
       output = model_fit.forecast()
       yhat = output[0]
       model_predictions.append(yhat)
       true_test_value = test_data[time_point]
       history.append(true_test_value)
   ```

5. Once the model is created we are going to predict the values for each test set. 

6. Pay attention to the parameter configuration used here which is (4, 1, 0) for the (p, d, q) values. 

7. After predicting, we will calculate the mean squared error for the actual price vs the predicted price.

   ```python
   MSE_error = mean_squared_error(test_data, model_predictions)
   print('Testing Mean Squared Error is {}'.format(MSE_error))
   ```


****

### Plotting the predictions 

8. For our clear understanding, we can plot the predicted values. 

   ```python
   test_set_range = data[int(len(data)*0.7):].index
   
   plt.plot(test_set_range, model_predictions, color='blue', marker='o',
            linestyle='dashed', label='Predicted Price')
   
   plt.plot(test_set_range, test_data, color='red', label='Actual Price')
   
   plt.title('Facebook Price Prediction')
   plt.xlabel('Date')
   plt.ylabel('Prices')
   plt.legend()
   plt.show()
   ```


****

### Output and LSTM model

9. The LSTM model developed before also predicts the price for the stocks and both we can print out both the predictions. These predictions can then be used any way we want. 

The output for error calculation and plotting the predicted price is below, now we will configure different parameters with the aim of decreasing the error. 
![output](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Machine learning 3\output.PNG)

![prediction_using_ARIMA](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Machine learning 3\prediction_using_ARIMA.PNG)



****

### Hyperparameter configurations

Currently we are using the configuration

```python
(p=4, d=1, q=0)
```

![error_(4,1,0)](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Machine learning 3\error_(4,1,0).PNG)

10. We can manually change these configurations and choose the config that produces the least error. 

11. Different configurations and the errors. We will start by decreasing the `p` value. 

    ```python
    (p=1, d=1, q=0)
    ```

    ![error_(1,1,0)](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Machine learning 3\error_(1, 1, 0).PNG)

The error for this config decreases by 0.65%. 

12. Now, we can change the `d` values.

    ```py
    (p=1, d=3, q=0)
    ```

    ![error_(1,3,0)](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Machine learning 3\error_(1, 3, 0).PNG)

The error for this config is too significant. 

13. Now we change the `q` values. 

    ```python
    (p=1, d=1, q=1)
    (p=1, d=1, q=3) 
    ```

    ![error_(1,1,1)](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Machine learning 3\error_(1, 1, 1).PNG)

    ![error_(1,1,3)](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Machine learning 3\error_(1, 1, 3).PNG)

    For both these configurations the difference in error is minimal. 

**output**: The least error prone configuration is (1, 1, 0) and hence we will use this configuration for our project. 

****

This concludes our task and we have successfully developed an ensemble approach to predicting the stock prices.   

