import numpy as np
import matplotlib.pyplot as plt
from DataExtraction2 import *
import DataExtraction2 as de

# FIELDS

# This is the font for the graphs
font = {'fontname' :  'Times New Roman'}

# FUNCTIONS

# This plots the particular data as a cmesh numpy plot
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

# MAIN FUNCTIONALITY

# This just specifies which bands we want to use
band_num = 5

de.bands = [band_num]

de.horiz_disp = 100
de.vert_disp = 10

# This gets the band data for each year
[band_data_1, band_data_2] = set_up_bands(y1_filename, y2_filename)

green_1 = band_data_1["B%d" % band_num]
green_2 = band_data_2["B%d" % band_num]

green_change = green_2 - green_1

titles = [
    "Visualizing Offsets Using Near Infrared Light (Band 5)",
    "West - East",
    "North - South"
]

plot_cmesh(green_change[3000:3500, 3000:3500], titles = titles, label = "TOA Reflectance")