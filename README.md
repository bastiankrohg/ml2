# Introduction

4A AE-SE Machine Learning TP 2
By Bastian Krohg et Nicolas Siard 

In this lab, we have learned to split data sets, train and tune a multi-layer perceptron. Our data sets are split into three parts: training set, test set and validation set. This allows us to train a multi-layer perceptron using different hyperparameters, and then test and validate our trained model and choice of tuning. 

The objective of this lab is to dive into particular kind of neural network: the *Multi-Layer Perceptron* (MLP) (slide 53 of the slides)

To start, let us take the dataset from the previous lab (hydrodynamics of sailing boats) and use scikit-learn to train a MLP instead of our hand-made single perceptron.
The code below is already complete and is meant to give you an idea of how to construct an MLP with scikit-learn. You can execute it, taking the time to understand the idea behind each cell.


```python
# Importing the dataset
import numpy as np
dataset = np.genfromtxt("yacht_hydrodynamics.data", delimiter='')
X = dataset[:, :-1]
Y = dataset[:, -1]
```


```python
# Preprocessing: scale input data 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
```


```python
# Split dataset into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=1, test_size = 0.20)
```


```python
# Define a multi-layer perceptron (MLP) network for regression
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(max_iter=3000, random_state=1) # define the model, with default params
mlp.fit(x_train, y_train) # train the MLP
```




<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>MLPRegressor(max_iter=3000, random_state=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">MLPRegressor</label><div class="sk-toggleable__content"><pre>MLPRegressor(max_iter=3000, random_state=1)</pre></div></div></div></div></div>




```python
# Evaluate the model
from matplotlib import pyplot as plt

print('Train score: ', mlp.score(x_train, y_train))
print('Test score:  ', mlp.score(x_test, y_test))
plt.plot(mlp.loss_curve_)
plt.xlabel("Iterations")
plt.ylabel("Loss")

```

    Train score:  0.9983128423378791
    Test score:   0.9973658105322623





    Text(0, 0.5, 'Loss')




    
![png](mlp_files/mlp_5_2.png)
    



```python
# Plot the results
num_samples_to_plot = 20
plt.plot(y_test[0:num_samples_to_plot], 'ro', label='y')
yw = mlp.predict(x_test)
plt.plot(yw[0:num_samples_to_plot], 'bx', label='$\hat{y}$')
plt.legend()
plt.xlabel("Examples")
plt.ylabel("f(examples)")
```




    Text(0, 0.5, 'f(examples)')




    
![png](mlp_files/mlp_6_1.png)
    


### Analyzing the network

Many details of the network are currently hidden as default parameters.

Using the [documentation of the MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html), answer the following questions.

- What is the structure of the network?
    - The MLPRegressor network is a neural network, by default it has one hidden layer containing 100 neurons. 
- What it is the algorithm used for training? Is there an algorithm available that we mentioned during the courses?
    - The training algorithm which is used is Adam. This is the solver for weight optimization of the neural network. The Adam algorithm is used for large datasets (thousands of training samples or more). 
    - 'sgd' or the "stochastic gradient descent" is an option which is an algorithm that we talked about in class. Adam is an optimized version of this approach. 
- How does the training algorithm decide to stop the training?
    - The algorithm stops iteration after:
        - having finished max number of iterations
        or 
        - convergence reached, determined by 'tol' which is by default 0.0001. This mecanism compares the current loss with tol, the tolerance for the optimization, and stops once the loss is less than tol. 

# Onto a more challenging dataset: house prices

For the rest of this lab, we will use the (more challenging) [California Housing Prices dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices).


```python
# clean all previously defined variables for the sailing boats
%reset -f
```


```python
"""Import the required modules"""
from sklearn.datasets import fetch_california_housing
import pandas as pd

num_samples = 2000 # only use the first N samples to limit training time

cal_housing = fetch_california_housing()
X = pd.DataFrame(cal_housing.data,columns=cal_housing.feature_names)[:num_samples]
y = cal_housing.target[:num_samples]

X.head(10) # print the first 10 values
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.3252</td>
      <td>41.0</td>
      <td>6.984127</td>
      <td>1.023810</td>
      <td>322.0</td>
      <td>2.555556</td>
      <td>37.88</td>
      <td>-122.23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.3014</td>
      <td>21.0</td>
      <td>6.238137</td>
      <td>0.971880</td>
      <td>2401.0</td>
      <td>2.109842</td>
      <td>37.86</td>
      <td>-122.22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2574</td>
      <td>52.0</td>
      <td>8.288136</td>
      <td>1.073446</td>
      <td>496.0</td>
      <td>2.802260</td>
      <td>37.85</td>
      <td>-122.24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.6431</td>
      <td>52.0</td>
      <td>5.817352</td>
      <td>1.073059</td>
      <td>558.0</td>
      <td>2.547945</td>
      <td>37.85</td>
      <td>-122.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.8462</td>
      <td>52.0</td>
      <td>6.281853</td>
      <td>1.081081</td>
      <td>565.0</td>
      <td>2.181467</td>
      <td>37.85</td>
      <td>-122.25</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.0368</td>
      <td>52.0</td>
      <td>4.761658</td>
      <td>1.103627</td>
      <td>413.0</td>
      <td>2.139896</td>
      <td>37.85</td>
      <td>-122.25</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.6591</td>
      <td>52.0</td>
      <td>4.931907</td>
      <td>0.951362</td>
      <td>1094.0</td>
      <td>2.128405</td>
      <td>37.84</td>
      <td>-122.25</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.1200</td>
      <td>52.0</td>
      <td>4.797527</td>
      <td>1.061824</td>
      <td>1157.0</td>
      <td>1.788253</td>
      <td>37.84</td>
      <td>-122.25</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.0804</td>
      <td>42.0</td>
      <td>4.294118</td>
      <td>1.117647</td>
      <td>1206.0</td>
      <td>2.026891</td>
      <td>37.84</td>
      <td>-122.26</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3.6912</td>
      <td>52.0</td>
      <td>4.970588</td>
      <td>0.990196</td>
      <td>1551.0</td>
      <td>2.172269</td>
      <td>37.84</td>
      <td>-122.25</td>
    </tr>
  </tbody>
</table>
</div>



Note that each row of the dataset represents a **group of houses** (one district). The `target` variable denotes the average house value in units of 100.000 USD. Median Income is per 10.000 USD.

### Extracting a subpart of the dataset for testing

- Split the dataset between a training set (75%) and a test set (25%)

Please use the conventional names `X_train`, `X_test`, `y_train` and `y_test`.


```python
# TODO
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1, test_size = 0.25)
```

### Scaling the input data


A step of **scaling** of the data is often useful to ensure that all input data centered on 0 and with a fixed variance.

Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance). The function `StandardScaler` from `sklearn.preprocessing` computes the standard score of a sample as:

```
z = (x - u) / s
```

where `u` is the mean of the training samples, and `s` is the standard deviation of the training samples.

Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Mean and standard deviation are then stored to be used on later data using transform.

 - Apply the standard scaler to both the training dataset (`X_train`) and the test dataset (`X_test`).
 - Make sure that **exactly the same transformation** is applied to both datasets.

[Documentation of standard scaler in scikit learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)




```python
# TODO
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()

#We calculate the 'global' mean which is used during the transformation
scaler.fit(X) 

#Transforming the two datasets to be centered on 0 and with a fixed variance
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Overfitting

In this part, we are only interested in maximizing the **train score**, i.e., having the network memorize the training examples as well as possible.

- Propose a parameterization of the network (shape and learning parameters) that will maximize the train score (without considering the test score).
    - We propose a neural network of 6 hidden layers of 100 neurons each. 

While doing this, you should (1) remain within two minutes of training time, and (2) obtain a score that is greater than 0.90.

- Is the **test** score substantially smaller than the **train** score (indicator of overfitting) ?
    - Yes, the test score is about 20% worse than the train score.
- Explain how the parameters you chose allow the learned model to overfit.
    - We chose to create a neural network containing more layers and neurons than what is necessary for our dataset. This way, the polynomial function the regressor creates to model the features of the dataset is too specific, meaning the test score will be substantially worse than the recognition rate of the training set since it is fitted to recognize small differences in the training set. 


```python
%%time 
#The "time" cell magic function allows us to visualize the time it takes to train and test our mlp
# TODO
# Defining a multi-layer perceptron (MLP) network for regression
from sklearn.neural_network import MLPRegressor

# define the model, with our params to maximize training score
mlp = MLPRegressor(max_iter=3000, hidden_layer_sizes=(100,100,100,100,100,100), random_state=1) 
mlp.fit(X_train_scaled, y_train) # train the MLP 

print(f"Train score: {mlp.score(X_train_scaled, y_train)}")
print(f"Test score: {mlp.score(X_test_scaled, y_test)}")
```

    Train score: 0.9430835811976644
    Test score: 0.7259472975709911
    CPU times: user 28.2 s, sys: 1.35 s, total: 29.6 s
    Wall time: 3.9 s


## Hyperparameter tuning

In this section, we are now interested in maximizing the ability of the network to predict the value of unseen examples, i.e., maximizing the **test** score.
You should experiment with the possible parameters of the network in order to obtain a good test score, ideally with a small learning time.

Parameters to vary:

- number and size of the hidden layers
- activation function
- stopping conditions
- maximum number of iterations
- initial learning rate value

Results to present for the tested configurations:

- Train/test score
- training time


Present in a table the various parameters tested and the associated results. You can find in the last cell of the notebook a code snippet that will allow you to plot tables from python structure.
Be methodical in the way your run your experiments and collect data. For each run, you should record the parameters and results into an external data structure.

(Note that, while we encourage you to explore the solution space manually, there are existing methods in scikit-learn and other learning framework to automate this step as well, e.g., [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html))


```python
# TODO
import pandas as pd
import numpy as np
import time

data = []


#training 0
start_training = time.time()
mlp = MLPRegressor(tol=0.00001,max_iter=2500, hidden_layer_sizes=(10, 10), random_state=1) #early_stopping: false and learning_rate_init: 0.001, activation: 'relu'
mlp.fit(X_train_scaled, y_train) # train the MLP 
stop_training = time.time()
#results 0
training_score = mlp.score(X_train_scaled, y_train)
test_score = mlp.score(X_test_scaled, y_test)
data.append({'tol': 10e-5, 'structure': 'hidden layers: 4, neurons:(100, 100, 100, 100)','activation': 'relu', 'learning_rate_init': '0.001', 'max_iter': '500', 'early_stopping': False, 'test_score': test_score, 'train_score': training_score, 'training_time': stop_training - start_training})

#training 1
start_training = time.time()
mlp = MLPRegressor(max_iter=2500, hidden_layer_sizes=(100, 100, 100, 100), random_state=1) #early_stopping: false and learning_rate_init: 0.001, activation: 'relu'
mlp.fit(X_train_scaled, y_train) # train the MLP 
stop_training = time.time()
#results 1
training_score = mlp.score(X_train_scaled, y_train)
test_score = mlp.score(X_test_scaled, y_test)
data.append({'structure': 'hidden layers: 4, neurons:(100, 100, 100, 100)','activation': 'relu', 'learning_rate_init': '0.001', 'max_iter': '2500', 'early_stopping': False, 'test_score': test_score, 'train_score': training_score, 'training_time': stop_training - start_training})

#training 2
start_training = time.time()
mlp = MLPRegressor(max_iter=250, hidden_layer_sizes=(10, 10), random_state=1) #early_stopping: false and learning_rate_init: 0.001, activation: 'relu'
mlp.fit(X_train_scaled, y_train) # train the MLP 
stop_training = time.time()
#results 2
training_score = mlp.score(X_train_scaled, y_train)
test_score = mlp.score(X_test_scaled, y_test)
data.append({'structure': 'hidden layers: 2, neurons:(10, 10)','activation': 'relu', 'learning_rate_init': '0.001', 'max_iter': '250', 'early_stopping': False, 'test_score': test_score, 'train_score': training_score, 'training_time': stop_training - start_training})

#training 3
start_training = time.time()
mlp = MLPRegressor(early_stopping=True, max_iter=2500, hidden_layer_sizes=(10, 10), random_state=1) #early_stopping: false and learning_rate_init: 0.001, activation: 'relu'
mlp.fit(X_train_scaled, y_train) # train the MLP 
stop_training = time.time()
#results 3
training_score = mlp.score(X_train_scaled, y_train)
test_score = mlp.score(X_test_scaled, y_test)
data.append({'structure': 'hidden layers: 2, neurons:(10, 10)','activation': 'relu', 'learning_rate_init': '0.001', 'max_iter': '2500', 'early_stopping': True, 'test_score': test_score, 'train_score': training_score, 'training_time': stop_training - start_training})

#training 4
start_training = time.time()
mlp = MLPRegressor(max_iter=5000, hidden_layer_sizes=(10, 10), random_state=1, learning_rate_init=0.005) #early_stopping: false and learning_rate_init: 0.001, activation: 'relu'
mlp.fit(X_train_scaled, y_train) # train the MLP 
stop_training = time.time()
#results 4
training_score = mlp.score(X_train_scaled, y_train)
test_score = mlp.score(X_test_scaled, y_test)
data.append({'structure': 'hidden layers: 2, neurons:(10, 10)','activation': 'relu', 'learning_rate_init': '0.001', 'max_iter': '5000', 'early_stopping': False, 'test_score': test_score, 'train_score': training_score, 'training_time': stop_training - start_training})

#training 5
start_training = time.time()
mlp = MLPRegressor(max_iter=5000, hidden_layer_sizes=(50), random_state=1) #early_stopping: false and learning_rate_init: 0.001, activation: 'relu'
mlp.fit(X_train_scaled, y_train) # train the MLP 
stop_training = time.time()
#results 5
training_score = mlp.score(X_train_scaled, y_train)
test_score = mlp.score(X_test_scaled, y_test)
data.append({'structure': 'hidden layers: 1, neurons:50','activation': 'relu', 'learning_rate_init': '0.001', 'max_iter': '5000', 'early_stopping': False, 'test_score': test_score, 'train_score': training_score, 'training_time': stop_training - start_training})

#training 6
start_training = time.time()
mlp = MLPRegressor(max_iter=2500, hidden_layer_sizes=(10,10,10), random_state=1, learning_rate_init=0.1) #early_stopping: false and learning_rate_init: 0.001, activation: 'relu'
mlp.fit(X_train_scaled, y_train) # train the MLP 
stop_training = time.time()
#results 6
training_score = mlp.score(X_train_scaled, y_train)
test_score = mlp.score(X_test_scaled, y_test)
data.append({'structure': 'hidden layers: 3, neurons:(10, 10, 10)','activation': 'relu', 'learning_rate_init': '0.1', 'max_iter': '2500', 'early_stopping': False, 'test_score': test_score, 'train_score': training_score, 'training_time': stop_training - start_training})

#training 7
start_training = time.time()
mlp = MLPRegressor(max_iter=500, hidden_layer_sizes=(10, 10), random_state=1, activation='identity') #early_stopping = False, learning_rate_init: 0.001, activation: 'relu'
mlp.fit(X_train_scaled, y_train) # train the MLP 
stop_training = time.time()
#results 7
training_score = mlp.score(X_train_scaled, y_train)
test_score = mlp.score(X_test_scaled, y_test)
data.append({'activation':'identity','structure': 'hidden layers: 2, neurons:(10, 10)', 'learning_rate_init': '0.001', 'max_iter': '500', 'early_stopping': False, 'test_score': test_score, 'train_score': training_score, 'training_time': stop_training - start_training})

#training 8
start_training = time.time()
mlp = MLPRegressor(max_iter=500, hidden_layer_sizes=(10, 10), random_state=1, activation='tanh') #early_stopping = False, learning_rate_init: 0.001, activation: 'relu'
mlp.fit(X_train_scaled, y_train) # train the MLP 
stop_training = time.time()
#results 8
training_score = mlp.score(X_train_scaled, y_train)
test_score = mlp.score(X_test_scaled, y_test)
data.append({'activation':'tanh','structure': 'hidden layers: 2, neurons:(10, 10)', 'learning_rate_init': '0.001', 'max_iter': '500', 'early_stopping': False, 'test_score': test_score, 'train_score': training_score, 'training_time': stop_training - start_training})

#training 9
start_training = time.time()
mlp = MLPRegressor(max_iter=500, hidden_layer_sizes=(13, 7), random_state=1) #early_stopping: false and learning_rate_init: 0.001, activation: 'relu'
mlp.fit(X_train_scaled, y_train) # train the MLP 
stop_training = time.time()
#results 9
training_score = mlp.score(X_train_scaled, y_train)
test_score = mlp.score(X_test_scaled, y_test)
data.append({'structure': 'hidden layers: 2, neurons:(13, 7)','activation': 'relu', 'learning_rate_init': '0.001', 'max_iter': '500', 'early_stopping': False, 'test_score': test_score, 'train_score': training_score, 'training_time': stop_training - start_training})

#training 10
start_training = time.time()
mlp = MLPRegressor(max_iter=250, hidden_layer_sizes=(10, 10), random_state=1, early_stopping=True) #learning_rate_init: 0.001, activation: 'relu'
mlp.fit(X_train_scaled, y_train) # train the MLP 
stop_training = time.time()
#results 10
training_score = mlp.score(X_train_scaled, y_train)
test_score = mlp.score(X_test_scaled, y_test)
data.append({'structure': 'hidden layers: 2, neurons:(10, 10)','activation': 'relu', 'learning_rate_init': '0.001', 'max_iter': '250', 'early_stopping': True, 'test_score': test_score, 'train_score': training_score, 'training_time': stop_training - start_training})


#Visualization of the data obtained from our tests
table = pd.DataFrame.from_dict(data)
table = table.replace(np.nan, '-')
table = table.sort_values(by='test_score', ascending=False)
table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tol</th>
      <th>structure</th>
      <th>activation</th>
      <th>learning_rate_init</th>
      <th>max_iter</th>
      <th>early_stopping</th>
      <th>test_score</th>
      <th>train_score</th>
      <th>training_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>-</td>
      <td>hidden layers: 2, neurons:(10, 10)</td>
      <td>relu</td>
      <td>0.001</td>
      <td>250</td>
      <td>False</td>
      <td>0.812961</td>
      <td>0.773523</td>
      <td>0.271912</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-</td>
      <td>hidden layers: 2, neurons:(10, 10)</td>
      <td>relu</td>
      <td>0.001</td>
      <td>2500</td>
      <td>True</td>
      <td>0.807264</td>
      <td>0.766763</td>
      <td>0.274380</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-</td>
      <td>hidden layers: 2, neurons:(10, 10)</td>
      <td>relu</td>
      <td>0.001</td>
      <td>250</td>
      <td>True</td>
      <td>0.807264</td>
      <td>0.766763</td>
      <td>0.282830</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-</td>
      <td>hidden layers: 3, neurons:(10, 10, 10)</td>
      <td>relu</td>
      <td>0.1</td>
      <td>2500</td>
      <td>False</td>
      <td>0.804720</td>
      <td>0.769171</td>
      <td>0.070125</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-</td>
      <td>hidden layers: 2, neurons:(13, 7)</td>
      <td>relu</td>
      <td>0.001</td>
      <td>500</td>
      <td>False</td>
      <td>0.802840</td>
      <td>0.801493</td>
      <td>0.369938</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-</td>
      <td>hidden layers: 2, neurons:(10, 10)</td>
      <td>tanh</td>
      <td>0.001</td>
      <td>500</td>
      <td>False</td>
      <td>0.796436</td>
      <td>0.809232</td>
      <td>0.378324</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.0001</td>
      <td>hidden layers: 4, neurons:(100, 100, 100, 100)</td>
      <td>relu</td>
      <td>0.001</td>
      <td>500</td>
      <td>False</td>
      <td>0.792111</td>
      <td>0.828632</td>
      <td>0.875639</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-</td>
      <td>hidden layers: 2, neurons:(10, 10)</td>
      <td>relu</td>
      <td>0.001</td>
      <td>5000</td>
      <td>False</td>
      <td>0.789289</td>
      <td>0.830669</td>
      <td>0.278828</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-</td>
      <td>hidden layers: 1, neurons:50</td>
      <td>relu</td>
      <td>0.001</td>
      <td>5000</td>
      <td>False</td>
      <td>0.787263</td>
      <td>0.821677</td>
      <td>0.652668</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-</td>
      <td>hidden layers: 4, neurons:(100, 100, 100, 100)</td>
      <td>relu</td>
      <td>0.001</td>
      <td>2500</td>
      <td>False</td>
      <td>0.771896</td>
      <td>0.944599</td>
      <td>4.501792</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-</td>
      <td>hidden layers: 2, neurons:(10, 10)</td>
      <td>identity</td>
      <td>0.001</td>
      <td>500</td>
      <td>False</td>
      <td>0.743002</td>
      <td>0.689359</td>
      <td>0.213180</td>
    </tr>
  </tbody>
</table>
</div>



In the table, we have visualized the manual iterations for which we have changed one or more parameters to see if test performance changed. We notice that our second iteration was the model with the best parameters. 

We also tried around 15 other combinations that performed worse than the iterations in the table, and we left these aside as we did not want our testing to take too much time. This additional testing allowed us to observe more of the impact of changing the different parameters, and thus tune the hyperparameters with greater precision.

## Evaluation

- From your experiments, what seems to be the best model (i.e. set of parameters) for predicting the value of a house?
    - The best model seems to be one with 2 hidden layers of 10 neurons each. This way, we achieve good training and test scores, without overfitting. We trained this model using 250 as the max number of iterations. We kept the default values for learning_rate_init (0.001), activation model ('relu'), tol (10e-4) and early_stopping (False). 
Unless you used cross-validation, you have probably used the "test" set to select the best model among the ones you experimented with.
Since your model is the one that worked best on the "test" set, your selection is *biased*.

In all rigor the original dataset should be split in three:

- the **training set**, on which each model is trained
- the **validation set**, that is used to pick the best parameters of the model 
- the **test set**, on which we evaluate the final model


Evaluate the score of your algorithm on a test set that was not used for training nor for model selection.




```python
# TODO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler()
#splitting dataset into three: Training Set (70%), Validation Set (15%) and Test Set (15%)
X_training, X_1, y_training, y_1 = train_test_split(X, y,random_state=1, test_size = 0.3) #training = 0.7
X_test, X_validation, y_test, y_validation = train_test_split(X_1, y_1, random_state=1, test_size=0.5)

#We calculate the 'global' mean which is used during the transformation
scaler.fit(X) 

#Transforming the two datasets to be centered on 0 and with a fixed variance
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_validation_scaled = scaler.transform(X_validation)


#training with our best model
start_training = time.time()
mlp = MLPRegressor(max_iter=250, hidden_layer_sizes=(10, 10), random_state=1) #early_stopping: false and learning_rate_init: 0.001, activation: 'relu'
mlp.fit(X_train_scaled, y_train) # train the MLP 
stop_training = time.time()
#results
training_score = mlp.score(X_train_scaled, y_train)
test_score = mlp.score(X_test_scaled, y_test)
validation = mlp.score(X_validation_scaled, y_validation)

print(f"Training score: {training_score}")
print(f"Test score: {test_score}")
print(f"Verification of model tuning - Validation score: {validation}")
print(f"Training time: {stop_training-start_training}")
```

    Training score: 0.7735227581042599
    Test score: 0.7907184010968576
    Verification of model tuning - Validation score: 0.8003574388307991
    Training time: 0.2170701026916504


## Conclusion
We notice that the validation score is about the same as both our test and our training score, which is a good sanity check that confirms that the model is well fitted and not only useful to memorize the data already used in training. The verification set scoring well indicates that our model is not too biased (i.e. underperforming with unknown data sets) and is a good performance indicator. 
