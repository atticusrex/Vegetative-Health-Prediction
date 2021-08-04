# IMPORTS 
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from PIL import Image
from DataExtraction2 import *
import DataExtraction2 as de
from keras.models import Sequential
from keras.layers import Dense
import keras

# DETAILS

__author__ = "Atticus Rex"
__copyright__ = "Copyright (C) 2021 Atticus Rex"
__license__ = "Public Domain"
__version__ = "1.0"

# FIELDS

# You have to change the maximum pixels or the library things its this defrag bomb or some crazy shit like that
Image.MAX_IMAGE_PIXELS = 1e9

# This is the font for the graphs
font = {'fontname' :  'Times New Roman'}


# FUNCTIONS

# This is going to be the function responsible for plotting the nice color mesh visuals
def plot_cmesh(image, titles = ["", "", ""], label = "", figure=1):
    # Unpacks the titles
    
    xlabel = titles[1] 
    ylabel = titles[2] 
    title = titles[0] 
    # Sets the correct coloring and labeling
    plt.figure(num=figure, figsize=(12, 10))
    plt.pcolormesh(image, cmap='nipy_spectral', vmin = -.2, vmax = 0.5)
    #plt.pcolormesh(image, cmap='RdYlGn')
    cbar = plt.colorbar()
    plt.title(title, font)
    plt.xlabel(xlabel, font)
    plt.ylabel(ylabel, font)
    cbar.ax.set_ylabel(label + "\n\n", font, rotation=270)
    plt.show()
    
# This function is useful for debugging and finding out the extreme values of an array
def get_extremes(array):
    min = 1e6
    max = -1e6
    for i in range(len(array)):
        if np.amax(array[i]) > max: max = np.amax(array[i])
        if np.amin(array[i]) < min: min = np.amin(array[i])
    
    return [max, min]

# This function gets one input for a particular set of coordinates
def get_one_input(band_data_1, row, col, width):
    # This is the list of inputs
    input_list = np.zeros((width * width, 10))
    # Sets up the class to scale the data into a 0-1 range
    scaler = MinMaxScaler()
        # Loops through each set of band data
    for i in range(1, 11):
        # Has to correct for excluding band 8 (Different size)
        if i > 7:
            # Copies the band data from the original photograph and saves it at the correct position in the input array.
            input_list[:, i - 1] = np.ravel(band_data_1["B%d" % (i + 1)][row:row + width, col:col + width])
        else:
            input_list[:, i - 1] = np.ravel(band_data_1["B%d" % (i)][row:row + width, col:col + width])
    

    # This scales the data into a 0-1 range
    for i in range(0, 10):
        input_list[:,i] = np.reshape(scaler.fit_transform(np.reshape(input_list[:, i], (-1, 1))), (len(input_list)))

    # Reshapes the data
    input_list = np.ravel(input_list)

    # Appends the date at the end 
    input_list = np.append(input_list, band_data_1["Date"].month / 12)

    return input_list

# This function needs to iterate through the subsets and use the model to predict them
def produce_predictions(model, band_data, subsets, width):
    sample_image = band_data["B1"]
    output_image = np.zeros(np.shape(sample_image))
    
    for subset in subsets:
        row = subset[0]
        col = subset[1]
        nn_input = np.array([get_one_input(band_data, row, col, width)])
        nn_output = model.predict(nn_input)
        prediction = nn_output[0][0]
        output_image[row:row + width, col:col + width] = prediction
    
    return output_image

def produce_full_predictions(model, band_data, mask, w):
    sample_image = band_data["B1"]
    output_image = np.zeros(np.shape(sample_image))

    # Finds the midpoint of the image
    row_segs = int(len(sample_image) / w) - 1
    col_segs = int(len(sample_image[0]) / w) - 1

    w

    for i in range(row_segs):
        row = i * w
        for j in range(col_segs):
            col = j * w
            if mask[row, col] != 0:
                nn_input = np.array([get_one_input(band_data, row, col, width)])
                nn_output = model.predict(nn_input)
                prediction = nn_output[0][0]
                output_image[row:row + width, col:col + width] = prediction
                print("Finished Row %d and Col %d" % (i, j))
    
    return output_image

def produce_current(band_data, subsets):
    ndvi = get_ndvi(band_data["B4"], band_data["B5"])
    output_image = np.zeros(np.shape(ndvi))

    for subset in subsets:
        row = subset[0]
        col = subset[1]
        
        output_image[row:row + width, col:col + width] = np.mean(ndvi[row:row + width, col:col + width])
    
    output_image[subset]

    return output_image

def load_model(name):
    json_file = open("Model\\" + name + ".json", 'r')
    model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(model_json)
    model.load_weights("Model\\" + name + ".h5")
    print("Loaded Model from Disk!")
    return model


model = load_model("NDVI_NDWI_19")

[band_data_1, band_data_2] = set_up_bands(y1_filename, y2_filename)

mask = create_mask(band_data_1["B1"], band_data_2["B1"])

#subsets = compile_subsets(band_data_1, band_data_2, mask)

ndvi = get_ndvi(band_data_1["B4"], band_data_1["B5"])

predictions = produce_full_predictions(model, band_data_1, mask, width)

change = predictions - ndvi

save_data("NNPredictions", predictions)
save_data("NNPredictedChange", change)