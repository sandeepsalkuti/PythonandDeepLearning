
from keras.models import Sequential                      #importing  sequential API style from keras
from keras.layers.core import Dense, Activation

# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd                                     #importing basic libraries
import numpy as np

dataset = pd.read_csv("diabetes.csv", header=None).values

X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,0:8], dataset[:,8],   
                                                    test_size=0.25, random_state=87)  #reading columns from dataset for train and test data
np.random.seed(155)                                                                   #using seed to customize start of random number generator
my_first_nn = Sequential()                                 # create model where sequential is linear stack of layers
my_first_nn.add(Dense(30, input_dim=8, activation='relu')) # hidden layer(2D-layer)
my_first_nn.add(Dense(40, input_dim=8, activation='relu')) # Adding multiple hidden layers
my_first_nn.add(Dense(50, input_dim=8, activation='relu')) # Adding multiple hidden layers
my_first_nn.add(Dense(20, input_dim=8, activation='relu')) # Adding multiple hidden layers

my_first_nn.add(Dense(1, activation='sigmoid'))            # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])#before training model configuring using learning process via compile method
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100,
                                     initial_epoch=0)      #training model
print("By adding more hidden layers performances results are:")
print(my_first_nn.summary())                               #summary prints complete description
print(my_first_nn.evaluate(X_test, Y_test))                #performance results
