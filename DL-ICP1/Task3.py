
from keras.models import Sequential          #importing  sequential API style from keras
from keras.layers.core import Dense

# load dataset
import pandas as pd
dataset = pd.read_csv("breastcancer.csv")    #loading dataset


X = dataset.iloc[:, 2:32].values
y = dataset.iloc[:, 1].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y) # Fit label encoder and return encoded labels M=1, B=0
print("y: ", y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#--------Data normalizes here---------#
from sklearn.preprocessing import StandardScaler     #applying standard scaling to normalize the data
sc = StandardScaler()

X_train = sc.fit_transform(X_train)                 
X_test = sc.transform(X_test)                         #after scaling transforming on test data

my_first_nn = Sequential() # create model where sequential is linear stack of layers
my_first_nn.add(Dense(20, input_dim=30, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #before training model configuring using learning process via compile method
my_first_nn_fitted = my_first_nn.fit(X_train, y_train, epochs=100, verbose=0,
                                     initial_epoch=0)   #training model

print("After Normalization performance resulyts are:\n")
print(my_first_nn.summary())                  #summary prints complete description
print(my_first_nn.evaluate(X_test, y_test))   #performance results




