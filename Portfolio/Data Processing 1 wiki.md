<h1><p align="center"> 
    Data Processing B.2
</h1>
**Name:** Shiva Gupta
**Student ID:** 102262514 
**Unit Code:** COS30018
**Task Number:** 2
**Repo link:** 

****
This report documents the process undertaken to process the data using desired parameters in a function. 

### Specifying the Dates for the Data set 

In the existing code base we have saved the dates and use them to load the data for training and testing, however we can do this in a load function and assign the existing values as default values so the code still runs as previous but it can be changed.

```C++
// Default values for the funtion 
TRAIN_START = dt.datetime(2012, 5, 23)     # Start date to read
TRAIN_END = dt.datetime(2020, 1, 7)       # End date to read

// Function definition 
def load_data(... , start_date = TRAIN_START, end_date = TRAIN_END); 

//function usages 
load_data() 
//function usage with dates
load_data(dt.datetime(2015, 8, 27), dt.datetime(2020, 8, 27))
```

1. The dates are then used to read the data from the `web.dataReader()` so we need to change it there. 

   ```C++
   data = web.DataReader(ticker, DATA_SOURCE, start_date, end_date)
   ```

2. These are two different outputs for different dates. 

![default output]()

![dates_changed]()

### Dealing with NAN 

[Resource: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html]
`NAN` means not a number, so we need to deal with the values that are not numbers in the data. We can use the `.dropna()` function provided in the pandas library in the DataFrame module. This function removes the missing values from the DataFrame object. 
The `implace=True` flag is true then it performs the function inplace and returns none. 

```C++
data = data = web.DataReader(ticker, DATA_SOURCE, start_date, end_date)
    ...

//Remove the NAN values
data.dropna(inplace = True)
```

### Split by Date 

Our Requirements state that we should be able to split the data in train/test sets based on dates or randomly. We can do this as such 

```C++

    if split_by_date:
        train_samples = int((1 - test_size) * len(x_train))
        x_train = x_train[:train_samples]
        y_train = y_train[:train_samples]

        x_test = x_train[train_samples:]
        y_test = y_train[train_samples:]
        if shuffle: 
            shuffle_in_unison(x_train, y_train)
            shuffle_in_unison(x_test, y_test)
    else: 
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size, shuffle=shuffle)
```

This function calculates the training sample size first based on the length of x_train list. Then it loops through the x_train list and assigns all index up to `training_samples` to x_train and all the indices after it to the x_test list. Similar functions is performed for y_test and y_train. 

Once the sets are distributed then it also uses the shuffle in unison function which shuffles the two lists randomly. 

```C++
def shuffle_in_unison(a, b): 
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)
```

If the split by date is false then it randomly distributes the training and test sets using the built-in train_test_split function. 

### Storing and Loading Data 

For our context the data is the data frame being read from the web using the pandas library. We are directly loading and using the data but we can load and store it locally using the function below

```C++
    if isinstance(ticker, str): 
        data = web.DataReader(ticker, DATA_SOURCE, start_date, end_date)
    elif isinstance(ticker, pd.DataFrame): 
        data = ticker
    else: 
        raise TypeError("ticker can be either a str or a 'pd.DataFrame' instance")
```

If the ticker is a string then we load the data and if the ticker is of the type of `pandas.DataFrame` then the data is the ticker that is of the type `pandas.DataFrame` . If both the instances are false then we raise a type error because ticker can only be of the type of string or `pd.DataFrame`. For our project we are using ticker of type string suck as 

```C++
COMPANY = 'FB'
    ...
def load_data(... , ticker = COMPANY, ...) 
    ...
```

### Scaling Feature Columns 

We are required to save the scaler that we are using to scale our data so that the scaler can be used later and also to scale our feature columns to include different types of prices like when the stock market is 'Open' etc. 
We can do this by using a list of string values for different feature prices like 

```C++
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Volume', 'Close', 'Adj Close']
    ...
def load_data(... , feature_columns = FEATURE_COLUMNS, ...) 
    ...
```

Now, for different feature columns we can also check if the data has columns for every feature price using the assert statement. 

```C+++
for col in feature_columns:
	assert col in data.columns, f"'{col}' does not exist in the dataframe."
```

![feature_columns]()
To store the scaler we can simply use a dictionary. 

```C++
//global scaler
scaler = MinMaxScaler(feature_range=(0, 1))

//use the scaler and assign it to the column_scaler
scaled_data = scaler.fit_transform(data[PRICE_VALUE].values.reshape(-1, 1))
column_scaler = scaler
```

![storing_scaler]()

Then this can be used later and we do not need to remember which scaler we used and we can extend the functionality to use multiple scalers. 

****

This concludes our task 2 for our project and we have successfully optimized processing and loading data. 