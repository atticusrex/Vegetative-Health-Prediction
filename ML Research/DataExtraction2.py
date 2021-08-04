# IMPORTS
import math
import numpy as np
from PIL import Image
import json
from pprint import pprint
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import gzip
import pickle
import os

# DETAILS


__author__ = "Atticus Rex"
__copyright__ = "Copyright (C) 2021 Atticus Rex"
__license__ = "Public Domain"
__version__ = "2.0"

# FIELDS

# Set this to 1 if you are running this program, 0 if you are running any other program that uses this file
running = 0

Image.MAX_IMAGE_PIXELS = 1e9

# Width of the section that we're inputting
width = 20

# This is where you enter the file names for the metadata files
y1_filename = "LC08_L1TP_017034_20200422_20200822_02_T1_MTL"
y2_filename = "LC08_L1TP_017034_20140524_20200911_02_T1_MTL"

# This is where you enter the displacement (in pixels) from one year to the next
# Essentially the distance you need to move the year 2 image (down = positive vert, right = positive horizontal)
horiz_disp = 140
vert_disp = 10

top_snip = 10
bottom_snip = 0
right_snip = 0
left_snip = 0

# This is something weird that happens with the numpy.load() function, these lines fix it
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# Specifies which bands we are looking for 
bands = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11]


# FUNCTIONS

# This gets the file details
def get_file_details(details_path):
    with open(details_path) as infile:
        details = json.load(infile)
    return details

# Converts a .TIF file to a numpy array
def tif_to_array(image_path):
    im = Image.open(image_path)
    imarray = np.array(im)
    imarray = np.array(imarray, dtype = np.float64)
    imarray = np.flip(imarray, axis = 0)
    
    return imarray

# Converts the file details dictionary into a dictionary of directories to make accessing each .TIF file much easier. 
def get_directory_names(file_details, bands):
    data_directory = {}
    directory_name = ""
    data_directory["Date"] = datetime.strptime(file_details['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['DATE_ACQUIRED'], '%Y-%m-%d')
    data_directory["B1"] = directory_name + file_details['LANDSAT_METADATA_FILE']['PRODUCT_CONTENTS']['FILE_NAME_BAND_1']
    data_directory["B2"] = directory_name + file_details['LANDSAT_METADATA_FILE']['PRODUCT_CONTENTS']['FILE_NAME_BAND_2']
    data_directory["B3"] = directory_name + file_details['LANDSAT_METADATA_FILE']['PRODUCT_CONTENTS']['FILE_NAME_BAND_3']
    data_directory["B4"] = directory_name + file_details['LANDSAT_METADATA_FILE']['PRODUCT_CONTENTS']['FILE_NAME_BAND_4']
    data_directory["B5"] = directory_name + file_details['LANDSAT_METADATA_FILE']['PRODUCT_CONTENTS']['FILE_NAME_BAND_5']
    data_directory["B6"] = directory_name + file_details['LANDSAT_METADATA_FILE']['PRODUCT_CONTENTS']['FILE_NAME_BAND_6']
    data_directory["B7"] = directory_name + file_details['LANDSAT_METADATA_FILE']['PRODUCT_CONTENTS']['FILE_NAME_BAND_7']
    data_directory["B8"] = directory_name + file_details['LANDSAT_METADATA_FILE']['PRODUCT_CONTENTS']['FILE_NAME_BAND_8']
    data_directory["B9"] = directory_name + file_details['LANDSAT_METADATA_FILE']['PRODUCT_CONTENTS']['FILE_NAME_BAND_9']
    data_directory["B10"] = directory_name + file_details['LANDSAT_METADATA_FILE']['PRODUCT_CONTENTS']['FILE_NAME_BAND_10']
    data_directory["B11"] = directory_name + file_details['LANDSAT_METADATA_FILE']['PRODUCT_CONTENTS']['FILE_NAME_BAND_11']
    data_directory["SE"] = float(directory_name + file_details['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_ELEVATION'])

    for band in bands:
        if band < 10:
            data_directory["M%d" % (band)] = float(directory_name + file_details['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_%d' % (band)])
            data_directory["A%d" % (band)] = float(directory_name + file_details['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_%d' % (band)])

    return data_directory

# This converts the raw values to top of atmosphere reflection :)
def convert_to_TOA_reflectance(image, paths, band_number):
    M = paths["M%d" % band_number]
    A = paths["A%d" % band_number]
    S = paths["SE"]

    return (M * image + A) / (math.sin(math.radians(S)))

# This gets the matrices for the bands
def get_band_data(year, paths, bands=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], Bottom=None):
    band_data = {"Date":paths["Date"]}
    if year == 1:
        for i in bands:
            image = tif_to_array("Data/Year1/" + paths["B%d" % i])
            if i < 10:
                image = convert_to_TOA_reflectance(image, paths, i)
            band_data["B%d" % i] = image
            
            print("Year 1 Band %d data acquired. . . " % (i))
    if year == 2:
        for band in bands:
            image = tif_to_array("Data/Year2/" + paths["B%d" % band])

            
            if band < 10:
                image = convert_to_TOA_reflectance(image, paths, band)
                
            band_data["B%d" % band] = correct_for_distortion(Bottom["B%d" % band], image, vert_disp, horiz_disp)
            print("Year 2 Band %d data acquired. . . " % (band))
    
    return band_data

# This sets up band data and aligns the two images
def set_up_bands(y1_filename, y2_filename):
    

    # This gets the file details for each year
    year1_details = get_file_details("Data/Year1/" + y1_filename + ".json")
    year2_details = get_file_details("Data/Year2/" + y2_filename + ".json")
    print("File Details Acquired")

    # This converts the file details into a more digestible dictionary format
    y1_paths = get_directory_names(year1_details, bands)
    y2_paths = get_directory_names(year2_details, bands)
    print("Paths Created")

    # This creates a numpy array from the .tif files for each of the photos (Band 1)
    band_data_1 = get_band_data(1, y1_paths, bands=bands)
    band_data_2 = get_band_data(2, y2_paths, bands=bands, Bottom=band_data_1)
    print("\n*********************\nAll Band Data Acquired!\n*********************\n")

    return [band_data_1, band_data_2]


# This creates a mask to filter out all of the dead areas in the photographs (Use Band 4 for best results)
def create_mask(image_array1, image_array2):
    mask1 = np.zeros(np.shape(image_array1))
    mask2 = np.zeros(np.shape(image_array2))
    mask1[image_array1 != 0] = 1
    mask1[mask1 != 1] = 0
    mask2[image_array2 != 0] = 1
    mask2[mask2 != 1] = 0

    mask1[mask2 == 0] = 0

    return np.array(mask1)

# This is going to break up a raw Landsat image into specific subsets
def create_subsets(image, mask, cloud_mask, water_mask, square_width):
    # This checks individual rectangles to see if they are in the mask or not
    def check_square(row, column, width):
        cloud_matrix = cloud_mask[row:row + width, column:column + width]
        #print("Mean: " + str(np.mean(cloud_matrix)) + " Max: " + str(np.amax(cloud_matrix)))
        cloud_threshold = 0.15 # This is the threshold to reject the presence of a cloud in a scene
        water_threshold = 0.25 # This threshold rejects the presence of a body of water in a scene
        if (0 in mask[row:row + width, column: column + width]):
            return False
        if (np.mean(cloud_mask[row:row + width, column:column + width]) > cloud_threshold):
            return False
        if (np.mean(water_mask[row:row + width, column:column + width]) < water_threshold):
            return False
        else:
            return True
                
    # This just renames the parameters to make the code less wordy
    im = image
    w = square_width

    # Finds the midpoint of the image
    row_segs = int(len(im) / w) - 1
    col_segs = int(len(im[0]) / w) - 1

    # Creates the list of possible subsets
    subsets = []

    for i in range(row_segs):
        row = i * w
        for j in range(col_segs):
            col = j * w
            if check_square(row, col, w):
                subsets.append([row, col])
    
    return subsets

# This corrects for any distortion encountered between the two photographs
def correct_for_distortion(bottom_img, top_img, vert_disp, horizontal_disp):
    if vert_disp < 0:
        vert_disp = abs(vert_disp)
        top_img = np.delete(top_img, slice(0, vert_disp, 1), axis=0)
        top_img = np.append(top_img, np.zeros((vert_disp, len(top_img[0]))), axis = 0)
    elif vert_disp > 0:
        top_img = np.delete(top_img, slice(len(top_img) - vert_disp, len(top_img)), axis = 0)
        top_img = np.concatenate((np.zeros((vert_disp, len(top_img[0]))), top_img), axis = 0)
    if horizontal_disp < 0:
        horizontal_disp = abs(horizontal_disp)
        top_img = np.delete(top_img, slice(0, horizontal_disp, 1), axis=1)
        top_img = np.append(top_img, np.zeros((len(top_img), horizontal_disp)), axis = 1)
    elif horizontal_disp > 0:
        top_img = np.delete(top_img, slice(len(top_img[0]) - 1 - horizontal_disp, len(top_img[0]) - 1), axis = 1)
        top_img = np.concatenate((np.zeros((len(top_img), horizontal_disp)), top_img), axis = 1)
    
    top_img = trim_image(top_img)

    return top_img

def trim_image(top_img):
    if top_snip != 0:
        top_img = np.delete(top_img, slice(0, top_snip, 1), axis=0)
    if bottom_snip != 0:
        top_img = np.delete(top_img, slice(len(top_img) - 1 - bottom_snip, len(top_img) - 1), axis = 0)
    if right_snip != 0:
        top_img = np.delete(top_img, slice(len(top_img[0]) - 1 - vert_disp, len(top_img[0]) - 1), axis = 1)
    if left_snip != 0:
        top_img = np.delete(top_img, slice(0, left_snip, 1), axis=1)
        
    return top_img

# Compress images more accurately
def compress_image(image, compression_scale):
    row_segs = int(len(image) / compression_scale)
    col_segs = int(len(image[0]) / compression_scale)

    new_img = np.zeros((row_segs, col_segs))

    for i in range(row_segs):
        for j in range(col_segs):
            row = i * compression_scale
            col = j * compression_scale
            
            new_img[i][j] = np.mean(image[row:row + compression_scale, col:col+compression_scale])

    return new_img

# This plots the particular data as a cmesh numpy plot
def plot_cmap(image, figure=1):
    image = np.array(image)
    plt.figure(num=figure, figsize=(12, 10))
    plt.pcolormesh(image, cmap='nipy_spectral')
    cbar = plt.colorbar()
    plt.show()

# This function crops a pixel image for zooming and viewing more specific datasets
def crop_image(image, width_range, height_range):
    w_min = width_range[0]
    w_max = width_range[1]
    h_min = height_range[0]
    h_max = height_range[1]

    new_img = []

    for i in range(len(image)):
        if (i <= h_max) and (i >= h_min):
            temp_list = []
            for j in range(len(image[i])):
                if (j <= w_max) and (j >= w_min):
                    temp_list.append(image[i][j])

            new_img.append(temp_list)
    
    return new_img

# This function displays the subsets on the graph of a particular band
def disp_subsets(image, subsets, width):
    image_mask = np.zeros(np.shape(image))
    for subset in subsets:
        row = subset[0]
        col = subset[1]
        for i in range(width):
            for j in range(width):
                image_mask[row][col] = -1
    image[image_mask == 0] = 0
    return image
    
# This function compares an input and output subsets and returns the list of subsets that they have in common
def filter_subsets(subset1, subset2):
    new_subsets = []

    for i in range(len(subset1)):
        if subset1[i] in subset2:
            new_subsets.append(subset1[i])
    
    return new_subsets

# This returns a numpy array with the NDVI calculated
def get_ndvi(band_4, band_5):
    sum_arr = band_4 + band_5
    diff_arr = band_5 - band_4
    
    # This filters out divide by zero errors of the mask
    sum_arr[sum_arr == 0] = 1
    diff_arr[sum_arr == 0] = 0

    ndvi = diff_arr / sum_arr 

    # This filters out abnormally high values from water
    ndvi[ndvi > 1] = 1

    return ndvi
    
# This returns a numpy array with the EVI calculated
def get_evi(band_2, band_4, band_5, band_6):
    numerator = band_5 - band_4
    denominator = (band_5 + 6 * band_4 - 7.5 * band_2 + 1)
    denominator[denominator == 0] = 1
    numerator[denominator == 0] = 0

    evi = 1 - 2.5 * numerator / denominator # NOTE: for some reason, the values on the website don't add 1, but you definitely do have to add 1 to make the calculations work

    
    evi[evi > 2] = 2
    evi[evi < -2] = -2
    

    return evi

# This returns a numpy array with the NDWI calculated
def get_ndwi(band_5, band_6):
    numerator = band_5 - band_6
    denominator = band_5 + band_6
    denominator[denominator == 0] = 1
    numerator[denominator == 0] = 0

    ndwi = numerator / denominator * 2.5

    ndwi[ndwi > 1] = 1
    ndwi[ndwi < 0] = 0

    return ndwi

# This compiles subsets of possible NN input data based upon a sepecified 
def compile_subsets(band_data_1, band_data_2, mask):
    print("Creating Subsets . . . ")
    # Now we have to create the different subsets for the data
    subsets_1 = create_subsets(band_data_1["B1"], mask, band_data_1["B4"], band_data_1["B5"], width)
    print("Subset 1 Collected! (Length: %d)" % (len(subsets_1)))

    subsets_2 = create_subsets(band_data_2["B1"], mask, band_data_2["B4"], band_data_2["B5"], width)
    print("Subset 2 Collected! (Length: %d)" % (len(subsets_2)))
    print("Individual Subsets Created\n")

    # Now we have to compare the subsets to each other and see how many overlap
    combined_subsets = filter_subsets(subsets_1, subsets_2)
    print("Combined Subsets Created (Length: %d)\n" % (len(combined_subsets)))
    return combined_subsets

# this will be responsible for creating conditioned inputs to the NN
def create_nn_inputs(band_data_1, compiled_subsets):
    print("Creating Inputs...")
    # This is the list of inputs
    input_list = np.zeros((len(compiled_subsets), width * width, 10))
    # Sets up the class to scale the data into a 0-1 range
    scaler = MinMaxScaler()

    # Loops through each subset to condition
    for index in range(len(compiled_subsets)):
        subset = compiled_subsets[index]
        # Loops through each set of band data
        for i in range(1, 11):
            # Has to correct for excluding band 8 (Different size)
            if i > 7:
                # Copies the band data from the original photograph and saves it at the correct position in the input array.
                input_list[index,:, i - 1] = np.ravel(band_data_1["B%d" % (i + 1)][subset[0]:subset[0] + width, subset[1]:subset[1] + width])
            else:
                input_list[index,:, i - 1] = np.ravel(band_data_1["B%d" % (i)][subset[0]:subset[0] + width, subset[1]:subset[1] + width])


    # This scales the data into a 0-1 range
    for i in range(0, 10):
        scaler.fit(input_list[:,:,i])
        input_list[:,:,i] = scaler.transform(input_list[:,:,i])

    # Reshapes the data so it isn't 3-dimensional
    input_list = np.reshape(input_list, (len(input_list), len(input_list[0]) * len(input_list[0][0])))

    # Appends the date to the input list

    dates = np.full((len(input_list), 1), band_data_1["Date"].month / 12)
    input_list = np.append(input_list, dates, axis=1)
    print("Inputs Created!")
    return input_list

def create_full_outputs(compiled_subsets, indicator):
    output_list = np.zeros((len(compiled_subsets), width * width))

    for i in range(len(compiled_subsets)):
        subset = compiled_subsets[i]

        output_list[i] = np.ravel(indicator[subset[0]:subset[0] + width, subset[1]:subset[1] + width])
    
    return output_list

# This creates the desired outputs for the neural network
def create_outputs(compiled_subsets, indicators):
    output_list = np.zeros((len(compiled_subsets), 2))

    for i in range(len(compiled_subsets)):
        subset = compiled_subsets[i]

        # This finds the mean of each of the three indicators within each subset
        output_list[i, 0] = np.mean(indicators[0][subset[0]:subset[0] + width, subset[1]:subset[1] + width])
        output_list[i, 1] = np.mean(indicators[1][subset[0]:subset[0] + width, subset[1]:subset[1] + width])
    
    print("Created Outputs!\n")
    return output_list

# This saves the conditioned inputs to a .npz (binary) file 
def save_data(file_name, data):
    print("Saving " + file_name + " data. . .")
    np.savez_compressed(file_name + '.npz', data)

# This appends the data to an existing file or creates one if there is none
def append_data(file_name, new_data, append=True):
    if os.path.isfile(file_name + '.npz') and append == True:
        current_data = load_data(file_name)
        current_data = np.append(current_data, new_data, axis=0)
        save_data(file_name, current_data)
    else:
        save_data(file_name, new_data)

# This loads the conditioned inputs from a .npz file
def load_data(file_name):
    dict_data = np.load(file_name + '.npz')
    data = dict_data['arr_0']
    return np.array(data)

# SCRIPT - This is the main functionality of the program

def main():
    # This gets the band data for each year
    [band_data_1, band_data_2] = set_up_bands(y1_filename, y2_filename)


    # Now we have to create a mask for the dead spots
    mask = create_mask(band_data_1["B1"], band_data_2["B1"])

    # This creates a combined subset list of everything we have
    combined_subsets = compile_subsets(band_data_1, band_data_2, mask)

    # This gets the indicator data for the second year (which is what we are targetting)
    ndvi = get_ndvi(band_data_2["B4"], band_data_2["B5"])
    ndwi = get_ndwi(band_data_2["B5"], band_data_2["B6"])
    print("Indicators Gathered!\n")

    indicators = [ndvi, ndwi]

    # This creates the inputs and corresponding outputs for the NN
    input_list = create_nn_inputs(band_data_1, combined_subsets)
    output_list = create_outputs(combined_subsets, indicators)

    # This saves the inputs and outputs to a binary compressed file
    append_data('Data\\NN Data\\inputs', input_list, append = False)
    append_data('Data\\NN Data\\outputs', output_list, append = False)


if running:
    main()