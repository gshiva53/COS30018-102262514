<h1><p align="center"> 
    Machine learning 1
</h1>

**Name:** Shiva Gupta
**Student ID:** 102262514 
**Unit Code:** COS30018
**Task Number:** 4
**Repo link:** 

****

This report documents the steps taken to refactor the code that build the machine learning model. 

### Arguments for Creating the model 

There are multiple arguments needed for building the model like the number of layers, type of network, length of the sequence etc. In the current code we are building a layer then adding the dropout and then building another layer and again adding the dropout. This process can be automated in the code. Also, the current network type is LSTM. 

1. Create a method and list all the arguments that are needed to build the model. Here we are creating a skeleton method in which we will later implement building the model. 

   ```python
   def create_model(sequence_length, n_features, units, cell, n_layers, dropout,
                    loss, optimizer, bidirectional):
       ...
       
   create_model()
   ```

2. We do not want the user to pass every value every time we want to build the model so we can define some default parameters. 

   ```python
   def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                   loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
           ...
       
   create_model()
   ```



### Building the Layers 

3. Now, we move on to building and adding the layers to model. previously we had to do this as such 

   ```python
   # First layer 
   model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
   model.add(Dropout(0.2))
   # second layer 
   model.add(LSTM(units=50, return_sequences=True))
   model.add(Dropout(0.2))
   # third layer
   model.add(LSTM(units=50))
   model.add(Dropout(0.2))
   model.add(Dense(units=1))  # Prediction of the next closest price
   ```

4. In the above code the arguments passed inside the `model.add()` are different but the statements are same so we can use simple `if` statements to automate this. 

5. The model is sequential and we create it first. Then for building the first layer we check if we want the layer to be bidirectional, if it is True then the first layer is a bidirectional layer otherwise it is a LSTM layer by default. 

   ```python
   model = Sequential()
       for i in range(n_layers):
           if i == 0:
               # first layer
               if bidirectional:
                   model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, 						sequence_length, n_features)))
               else:
                   model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, 
   ```

6. After we are done building the first layer then we build the next layer. Note that the number of layers are variable so we are building the first layer by ourselves and the last layers will all be of the same type

7. For last layer or layers because all the layers will be of the same type after first layer so 

   ```python
           elif i == n_layers - 1:
               # last layer
               if bidirectional:
                   model.add(Bidirectional(cell(units, return_sequences=False)))
               else:
                   model.add(cell(units, return_sequences=False))
   ```

8. We are also going to add a hidden layer to the model which will have the return_sequences argument set to true. 

   ```python
         else:
             # Hidden layers
             if bidirectional:
                 model.add(bidirectional(cell(units, return_sequences=True))
             else:
                 model.add(cell(units, return_sequences=True))
   ```

9. After every layer we are going to add a dropout layer. 

   ```python
    model.add(Dropout(dropout))
   ```

10. Then we will compile the model 

    ```python
    model.compile(optimizer=optimizer, loss=loss)
    ```

11. Then we return the model back to the caller and this way we have automated the model building process using some extra arguments. 

12. Call the create_model function and fix that model to predict the stock prices. 

    ```python
    model = create_model(50, 1)
    model.fit(x_train, y_train, epochs=3, batch_size=32)
    ```

![output]()

![graph]()



This concludes our task and we have successfully automated the model building process. 