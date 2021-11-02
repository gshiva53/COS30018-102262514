<h1><p align="center"> 
    Extension
</h1>

**Name:** Shiva Gupta
**Student ID:** 102262514 
**Unit Code:** COS30018
**Task Number:** 7
**Repo link:** 

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

This concludes our task and we have successfully developed an ensemble approach to predicting the stock prices.   

