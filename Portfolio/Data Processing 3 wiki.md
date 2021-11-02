<h1><p align="center"> 
    Data Processing B.3
</h1>

**Name:** Shiva Gupta
**Student ID:** 102262514 
**Unit Code:** COS30018
**Task Number:** 2
**Repo link:** 

****

This report documents the process undertaken to plot the data using desired arguments in a function. 

### Setting up the environment to use plotly module 

1. install the `plotly` module

   ```python
   $ pip install plotly
   ```

2. import the module in your project 

   ```python
   $ import plotly.graph_objects as go
   ```

### Understanding and implementing traces 

3. `trace` can be thought of as the options for plotting graphs. Multiple traces can be used consecutively within the same graph. 

4. First, we will set up the trace for the candlestick graph

   ```python
       trace1 = {
           'x': data.index,
           'open': data["Open"],
           'close': data["Close"],
           'high': data['High'],
           'low': data['Low'],
           'type': 'candlestick',
           'name': company,
           'showlegend': True
       }
   ```

5. The `x` stands for data to be plot on the x-axis. `Open`, `close`, `high` and `low` are the stock prices. `type` is the type of graph that we want and the `name` refers to the name of the graph. `showlegend` will show the legend for the graph. These are all the options for the candlestick trace. 

This is the output for the candlestick chart
![candlestick_output]()

6. The requirements state that we need to express the data for variable trading days and this can be achieved by plotting the average for specified days. 

   ```python
   # avg_window is the input the argument to the function
   avg = data['Close'].rolling(window=avg_window, min_periods=1).mean()
   ```

7. This `avg` can be plot in the y-axis and it can display the **moving averages**.

   ```python
   trace2 = {
           'x': data.index,
           'y': avg,
           'type': 'scatter',
           'mode': 'lines',
           'line': {
               'width': 1,
               'color': 'blue'
           },
           'name': company,
           'showlegend': True
       }
   ```

8. Plot the graph

   ```python
   fig = go.Figure(data=[trace1, trace2])
   fig.show()
   ```

Here is the output for both the traces plotted on the same graph. 

![output]() 

9. All these can be wrapped inside a function 

   ```python
   def plot_graph(avg_window):
       ...
       
       
   plot_graph(45)
   ```

10. The function can be called with a variable number of days and it will plot the averages as needed. 

Here is the difference between 45 and 90 days. 

![output_90]()

![output]()

This concludes our task of data processing for stock prediction. 