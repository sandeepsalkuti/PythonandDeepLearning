
from keras.models import Sequential
from keras.layers.core import Dense

# load dataset
import pandas as pd
dataset = pd.read_csv("breastcancer.csv")


X = dataset.iloc[:, 2:32].values
y = dataset.iloc[:, 1].values

print(dataset.iloc[:, 1].value_counts())

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y) # Fit label encoder and return encoded labels M=1, B=0
print("y: ", y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


my_first_nn = Sequential() # create model
my_first_nn.add(Dense(20, input_dim=30, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
my_first_nn_fitted = my_first_nn.fit(X_train, y_train, epochs=100, verbose=0,
                                     initial_epoch=0)

print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, y_test))

# --------------Q.3 Data is normalize here ----------------------#
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("\n----------Q.3 After Normalization---------------\n")
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, y_test))