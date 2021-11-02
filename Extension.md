<h1><p align="center"> 
    Extension
</h1>

**Name:** Shiva Gupta
**Student ID:** 102262514 
**Unit Code:** COS30018
**Task Number:** 7
**Repo link:** https://github.com/gshiva53/COS30018-102262514/tree/main/pythonProject

****

This report documents the approach undertaken fore extending the current stock prediction model to predict companies' trends and the steps taken to implement it. 

### Relative Strength Index(RSI)

[Source: https://www.investopedia.com/terms/r/rsi.asp]

Relative Strength Index (RSI) measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock. The RSI is displayed as a line graph that moves between two extremes reading from 0 to 100. An RSI of 70 or above indicate that security is becoming overbought or overvalued and a reading of 30 or below indicates an oversold or undervalued condition. 
RSI as an indicator is usually mapped on the same graph as the data while sharing the same x-axis. It is a popular momentum oscillator indicator and signals about the bullish and bearish price momentum. 

![Sample_Image_RSI](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Extension\Sample_image_RSI.webp)

Apart from reading overbought and oversold conditions, there are special features in RSI which can be used extended and used with other indicators. Some of the indicators that work with RSI are as follows. 

[Source: https://www.elearnmarkets.com/blog/5-roles-relative-strength-index-rsi/]

1. Trendline Application

1. Pattern Breakout 
2. Advance breakout and breakdown
3. Role of 50 
4. Failure Swing

Considering these features and the ease of implementation of the RSI indicator in our current prediction model it becomes an easy choice to implement this indicator and better our existing model. 


****

### RSI Calculation

[Source: https://www.elearnmarkets.com/blog/5-roles-relative-strength-index-rsi/]

Formula for the RSI index is as follows. 

```
RSI = 100 - 100 / (1 + RS)
RS = (Average gain over specified period) / (Average loss over the same period)
```

The default number of days measured for RSI is `14` but this can be changed based on the sensitivity of our equipment. 

For Example, low-default RSI is more sensitive to overbought and oversold conditions than high-default RSI 

****



### Implementing the RSI indicator

[Source: https://www.youtube.com/watch?v=oiheV1xXEtg]

We want to calculate the relative strength of the `Adjusted Close` values of the data. The data here refers to the stock prices that we are taking from the `Yahoo` database. We can use the same unscaled data and perform calculations on the adjusted close values. 

1. Load the data

2. Calculate the relative values of the adjusted close price of the stock.
   For this, we calculate the difference between two rows except the `NaN` values.

   ```python
   delta = data['Adj Close'].diff(1)
   delta.dropna(inplace=True)
   ```
   
3. We need the positive and negative values compared to the previous values of the stocks so that we can calculate the average loss and average gain. 
   
   ```python
   positive = delta.copy()
   negative = delta.copy()
   ```
   
4. Since, we need the relative values, and the existing calculations would produce values that are negative so we only need those values for the positive which are greater than zero and this logic applies to the negative values as well. We only need the negative values that affect the negative so the remaining values can just be 0. 

   ```python
   positive[positive < 0] = 0
   negative[negative > 0] = 0
   ```

5. Now, we can calculate the `average loss` and `average gain` for a time period. The default value for this time period is 14 but we are calculating the RSI values in a function so we can change the value for this time period easily. 

   ```python
   # low-default is more sensitive to overbought and oversold conditions than high-default
   days = days  # default is 14
   
   average_gain = positive.rolling(window=days).mean()
   average_loss = abs(negative.rolling(window=days).mean())
   ```

6. Now, we calculate the actual RSI value for every row. The formula for RSI is given above. 

   ```python
   relative_strength = average_gain / average_loss
   RSI = 100.0 - (100.0 / (1.0 + relative_strength))
   ```

7. These values can then be combined in a data frame and later be used to plot the values. 

   ```python
   combined = pd.DataFrame()
   combined['Adj Close'] = data['Adj Close']
   combined['RSI'] = RSI
   ```


****



### Plotting the Values

We move to plotting values using the `matplot.lib`. We need to plot the `Adjusted Close` values for the stocks with their respective RSI indicator just like in the sample image shown above. Both the graphs share the `x` axis for dates. We also need the guiding horizontal lines to indicate overbought and oversold stocks. 
We begin by plotting the adjusted close values. 

1. We plot the combined data frame's index versus the adjusted close values and just for aesthetics we set the face color to black with white values and light gray lines. 

   ```python
   # plotting the Adjusted Close value
   plt.figure(figsize=(12, 8))
   ax1 = plt.subplot(211)
   ax1.plot(combined.index, combined['Adj Close'], color='lightgray')
   ax1.set_title("Adjusted Close Price", color='white')
   
   ax1.grid(True, color='#555555')
   ax1.set_axisbelow(True)
   ax1.set_facecolor('black')
   ax1.figure.set_facecolor('#121212')
   ax1.tick_params(axis='both', colors='white')
   plot.show()
   ```

   ![Adjusted_Close_Price](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Extension\Adjusted_Close_Price.png)

2. Next we need to plot the RSI values with similar aesthetics with the stock price values. 

   ```python
       # plotting the RSI indicator
       ax2 = plt.subplot(212, sharex=ax1)
      	# -------- plot the lines for RSI indicator ------
       ax2.set_title("RSI Value")
       ax2.grid(False)
       ax2.set_axisbelow(True)
       ax2.set_facecolor('black')
       ax2.tick_params(axis='both', colors='white')
   
       plt.show()
   ```

   ![RSI_Values_without_hlines](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Extension\RSI_Values_Without_hlines.png)

   3. The horizontal line indicators can be plotted using the `axhline()` function. 

      ```python
      ax2.axhline(0, linestyle='--', alpha=0.5, color='#ff0000')
      ax2.axhline(10, linestyle='--', alpha=0.5, color='#ffaa00')
      ax2.axhline(20, linestyle='--', alpha=0.5, color='#00ff00')
      ax2.axhline(30, linestyle='--', alpha=0.5, color='#cccccc')
      ax2.axhline(70, linestyle='--', alpha=0.5, color='#cccccc')
      ax2.axhline(80, linestyle='--', alpha=0.5, color='#00ff00')
      ax2.axhline(90, linestyle='--', alpha=0.5, color='#ffaa00')
      ax2.axhline(100, linestyle='--', alpha=0.5, color='#ff0000')
      ```

      ![output](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Extension\output.png)

****

This concludes our task and we have successfully extended the stock prediction model to include an RSI indicator that can represent the companies' trends. 

