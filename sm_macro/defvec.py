# Macro which define bands in input for plotsed_all.sm
# It is the first macro to read
# Define vectors/bands

from astropy.io import fits
import numpy as np
import pandas as pd


# Solution of reading the file automatically and making array with filters
data = fits.getdata("stripe82_totalmag_1007_truncated.fits")
data_table = pd.DataFrame(data)
header = list(data_table.columns)
del header[:18]
del header[2]
del header[3:5]
del header[-1]
del header[-90:]
dataset_header = (header)
# dataset_header = dataset_header.pop(1)
print("--------- All filters are here ---------", dataset_header)
# removing A_ from the output list
# ch = 'A_'
# dataset_header = [elem.replace(ch, '') for elem in dataset_header]
# # making final array of the list
# dataset_header = pd.array(dataset_header)
print("WAVEBANDS")
print(dataset_header)

# Make array for central wavelengths of the bands(each of them) /// Hard to find wavelengths in Surveys web pages (need to be checked again)
central_values = data_table.iloc[0]
central_wavelengths = pd.to_numeric(central_values) # make all elements float
central_array = list(np.array(central_wavelengths))
del central_array[:18]
del central_array[2]
del central_array[3:5]
del central_array[-1]
del central_array[-90:]
vec_flux_obs = [item * 10000 for item in central_array]
print("CENTRAL WAVELENGTHS")
print(vec_flux_obs)

# Make another array for width of wavelengths of the bands(each of them) // Never found them(that's why some of them are not defined)
width_values = ["n/d", "n/d", "n/d", "n/d", "n/d", "n/d", "n/d", "n/d", "n/d", "n/d", "0.120", "0.213", "0.307", "0.390", "n/d", "n/d", "n/d", "n/d", "0.663", "1.042", "5.506"]
vec_fluxe_obs = np.array(width_values)
print("WIDTHS OF WAVELENGTHS")
print(vec_fluxe_obs)