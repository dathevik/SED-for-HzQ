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
del header[:13]
del header[-82:]
dataset_header = (header)
# removing A_ from the output list
ch = 'A_'
dataset_header = [elem.replace(ch, '') for elem in dataset_header]
# making final array of the list
bands_array = pd.array(dataset_header)
# one_band = bands_array[]   to take one filter if needed
print("WAVEBANDS")
print(bands_array)

# Make array for central wavelengths of the bands(each of them)
central_values = ["0.355", "0.468", "0.616", "0.748", "0.893", "0.243", "0.240", "0.234", "0.227", "0.214", "1.00", "1.250", "1.650", "2.150", "3.226", "11.332", "1.96", "11.326", "3.353", "4.603", "11.56"]
central_array = np.array(central_values)
print("CENTRAL WAVELENGTHS")
print(central_array)

# Make another array for width of wavelengths of the bands(each of them)
width_values = ["n/d", "n/d", "n/d", "n/d", "n/d", "n/d", "n/d", "n/d", "n/d", "n/d", "0.120", "0.213", "0.307", "0.390", "n/d", "n/d", "n/d", "n/d", "0.663", "1.042", "5.506"]
width_array = np.array(width_values)
print("WIDTHS OF WAVELENGTHS")
print(width_array)