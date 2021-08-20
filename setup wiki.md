<h1><p align="center"> 
    Setup B.1
</h1>
**Name:** Shiva Gupta
**Student ID:** 102262514 
**Unit Code:** COS30018
**Task Number:** 1
**Repo link:** 

****
This report documents the process undertaken to get projects **v0.1, p1 and p2** to run and understand the codebase for v0.1. **requirements.txt** files for every project have been attached with the project with the commands to setup the virtual environments. 

## Virtual Environment 

A virtual environment is a pseudo environment that contains dependent dependencies with the specified versions so that projects can have their own environments without sharing libraries and the developer will not have to switch between libraries and setup every time another project is opened. 
Virtual Environments: https://docs.python.org/3/tutorial/venv.html
The process to setup the environment and using it on windows with python 3.9 and pip 20.0 is documented below.  

1. To set up a virtual environment, we first install the python package 

```python
$ pip install virtualenv
```

![Installing virtual environment](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Week 1 - Setup\installing virtualenv.PNG)

1. Now, to use the virtual environments we first make a environment directory to keep all the environments in one place. 
2. To make our virtual environment we do: 

```python
$ python -m venv <name>					// conventional name is 'env' 
```

4. To use the environment in Windows platform

```python
$ <name>\Scripts\activate.bat
```

NOTE: Since we need to run a batch file so we need to use the `cmd`,  we can not use the `powershell`. 

5. To deactivate the virtual environment

```
>deactivate
```

****



## Running v0.1 - Stock_prediction.py

1. To check if pip is installed 

```python
$ python -m pip --verison
```

2. To install pip 

```python
$ sudo apt install python3-pip	
```

3. To check if pip is installed 

```python
$ pip3 --version 
```

4. Install the required libraries 

```python
$ pip install numpy 
$ pip install matplotlib 
$ pip install pandas 
$ pip install pandas-datareader
$ pip install tensorflow 
$ pip install scikit-learn 
```

5. Installing Libraries 

![Installing Libraries_1](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Week 1 - Setup\Installing Libraries_1.PNG)

![Installing Libraries_2](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Week 1 - Setup\Installing Libraries_2.PNG)



2. Running the code 

```python
>python ./stock-prediction.py
```

![v0.1_code_running](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Week 1 - Setup\v0.1_code_running.PNG)

****

## Understanding the codebase for v_0.1

For predicting the stock prices the code needs to train on the previous data and it uses `tensorflow` library for machine learning. It reads the data using `pandas-data reader`. It trains on the data using the sequential model and then plots a graph for actual and estimated values, after plotting it starts predicting the prices for next day. The plotting is done using the `matplotlib.pyplot`.  For the initial stages of the project it predicts the price for 'FaceBook'. 

****



## Running p1: Stock prediction using ML 

1. installing Libraries 

![installing libraries_1_p1](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Week 1 - Setup\installing libraries_1_p1.PNG)

![installing libraries_2_p1](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Week 1 - Setup\installing libraries_2_p1.PNG)

2. Change parameters in the 'parameters.py' file if needed. Here we change the `EPOCH` value from 500 to 50 just to test if the project runs in less time.

```python
EPOCH = 50 
```

2. Training p1

![training_p1_1](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Week 1 - Setup\training_p1_1.PNG)

![training_p1_2](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Week 1 - Setup\training_p1_2.PNG)

3. Running p1

![running_p1](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Week 1 - Setup\running_p1.PNG)





## Running p2

First, we need to create a separate virtual environment to run p2 because it uses older versions of tensorflow libraries. 

1. To create a virtual python environment and install the dependencies 

```python
$ bash init.sh	
```

The older versions of tensorflow libraries are not compatible with the new version of python so we need to uninstall and re-install  an older version. The accepted version is 3.7 or older. For keras the version is 2.2.4 that is compatible with tensorflow 1.15. 
The process to install the libraries and an older version of python is listed below. **Installing and Uninstalling python**

1. Go to Control Panel -> Uninstall a Program and delete the python version not required.
2. Install the version required using the official python website 
3. Delete the old PATH from the environment variables and add the new PATH. 

NOTE: We will have to create a new virtual env for the new python installation for it to work. 

 ```python
$ pip install mpl_finance
$ pip install mplfinance
$ pip install matplotlib
$ pip install pandas
$ pip install pandas-datareader
$ pip install tensorflow==1.15
$ pip install tensorflow-gpu==1.15
$ pip install fix_yahoo_finance
$ pip install arrow
$ pip isntall keras=2.2.4 // keras version 2.2.4 is compatible with 1.15
$ pip install sklearn 
 ```

After setting up the environment we move on to running the code. First we need to prepare the dataset meaning we need to train the code to predict prices using real data. The data is included in the project and the model can be trained using the command below. 

**Preparing Dataset Step 1**

```python
$ python runallfromlist.py tw50.csv 20 50
```

![preparing_dataset_step_1_p2](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Week 1 - Setup\preparing_dataset_step_1_p2.PNG)

After training the model we generate a data set using the command below. 
**Preparing Dataset Step 2**

```python
$ python generatebigdata.py dataset 20_50 bigdata_20_50
```

![preparing_dataset_step_2_p2.PNG](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Week 1 - Setup\preparing_dataset_step_2_p2.PNG)

The next logical step is to build the model and the command is

```python
$ python myDeepCNN.py -i dataset/bigdata_20_50
```

![Building the model](D:\Semester 2 2021\COS300018 - Intelligent Systems\COS30018-102262514\snips\Week 1 - Setup\building_the_model_p2.PNG)

