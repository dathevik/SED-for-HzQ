# Macro which reads the data in input for plotsed_all.sm
# It is the second macro to read all necessary files in the folder

import numpy as np
import pandas as pd
import os
import csv

# find data files from all folders and subfolders
# Get the list of all files in a directory and read them all
# make the data columns independent vectors

# I: Define the path and print a list of files from folders and subfolders
my_path = r'/home/dathev/PhD_Projects/4MOST/chicode_sm'
new_array = []  # make empty 1D array to collect files in the array
print("THIS IS THE LIST OF ALL FILES -------------")
# find all data files using loop
for root, subFolder, files in os.walk(my_path):
    for item in files:
        if item.endswith(".txt" or ".spc" or "dat" or "tab"):    #find files of the mentioned formates
                fileNamePath = np.array(os.path.join(root, item))
                mylist = (fileNamePath.tolist())
                b = new_array.append(mylist)    #make a list of files
                print(mylist)

# II: Use Pandas to make data of the files into a Dataframe
print(end="\n\n\n")
data_array = np.array(new_array)
read_file = data_array[1]
print("DATA INSIDE", read_file, "FILE")
# Using csv.reader to open csv files
with open(read_file, 'r') as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        data = pd.DataFrame(reader)
print(data)

# III: Make column-vector and write output in other file
column_vector = data[data.columns[1]]
print(column_vector)

# open new txt file
# new_f = open('/home/dathev/PhD_Projects/4MOST/chicode_sm/data_array.txt', "w")
# new_f.write(str(column_vector))


