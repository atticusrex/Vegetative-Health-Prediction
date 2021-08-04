# IMPORTS 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import keras
import matplotlib.pyplot as plt
from keras.optimizers import SGD
import random

# DETAILS

def legal():
    __author__ = "Atticus Rex"
    __copyright__ = "Copyright (C) 2021 Atticus Rex"
    __license__ = "Public Domain"
    __version__ = "1.0"

# FIELDS

# This has to modify the old np.load function (I have no idea how this works, stole this from stack overflow)
np_load_old = np.load

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# This is the font for the graphs
font = {'fontname' :  'Times New Roman'}

# FUNCTIONS

# This loads the conditioned inputs from a .npz file
def load_file(file_name):
    # This loads the data from the compressed .npz file
    dict_data = np.load(file_name + '.npz')
    data = dict_data['arr_0']
    return np.array(data)

def plot_loss(history):
    plt.title("Prediction Error vs. Epoch of Neural Network Prediction\n Final Accuracy: %f percent" % (accuracy), font)
    plt.xlabel("Number of Epochs", font)
    plt.ylabel("Error (%)")
    plt.plot(history.history['loss'], color='red')
    plt.grid(color='gray')
    plt.show()

# This function saves the model to a .json and .h5 file
def save_model(model, name):
    model_json = model.to_json()
    # Writes the model structure
    with open("Model\\" + name + ".json", 'w+') as json_file:
        json_file.write(model_json)
    # Saves the model's weights
    model.save_weights("Model\\" + name + ".h5")

    print("Saved Model to Disk!")

# This loads the model from an already-saved file
def load_model(name):
    # Opens the .json file
    json_file = open("Model\\" + name + ".json", 'r')
    model_json = json_file.read()
    json_file.close()

    # Produces a keras model from a json file
    model = keras.models.model_from_json(model_json)
    # Loads the weights from the .h5 file
    model.load_weights("Model\\" + name + ".h5")
    print("Loaded Model from Disk!")
    return model

# This function runs and trains the actual neural network
def run_model(X, Y):
    # Sequential model with 4 hidden layers 
    model = Sequential()
    model.add(keras.Input(shape=(len(X[0]))))
    model.add(Dense(1028, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation=None))

    # Compiling model
    opt = SGD(lr=.01, momentum=0.9, decay=0.005)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # Backpropagation
    history = model.fit(X, Y, epochs=3, batch_size=25)

    return [model, history]

# SCRIPT

def main():
    # Loads the input and output data
    X = load_file('Data\\NN Data\\inputs')
    Y = load_file('Data\\NN Data\\outputs')

    X_train = X[:85000]
    X_test = X[85000:]
    Y_train = Y[:85000]
    Y_test = Y[85000:]

    print(np.shape(X))
    print(np.shape(Y))

    # Runs the model
    [model, history] = run_model(X_train, Y_train)

    print(model.evaluate(X_test, Y_test))

    # Saves the Model
    save_model(model, "NDVI_NDWI_19")

main()