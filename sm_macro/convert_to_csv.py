import pandas as pd
import numpy as np
import os

#MAKE ARRAY OF ALL IMPORTANT FILES
my_path = r'/home/dathev/PhD_Projects/4MOST/chicode_sm'
new_array = []
for root, subFolder, files in os.walk(my_path):
    for item in files:
        if item.endswith(".spc") or item.endswith(".dat") or item.endswith(".tab") or item.endswith(".txt"):    #find files of the mentioned formates
                fileNamePath = np.array(os.path.join(root, item))
                mylist = (fileNamePath.tolist())
                new_array.append(mylist)

print(new_array)

#MAKE ANOTHER ARRAY OF ALL IMPORTANT FILES NAMES WITHOUT .txt or .dat or etc
name_array = []
for y in range(44):
    if new_array[y].endswith(".spc") or item.endswith(".dat") or item.endswith(".tab") or item.endswith(".txt"):
        m = new_array[y]
        size = len(m)
        new_names = np.array(m[:size-4])
        name_list = (new_names.tolist())
        name_array.append(name_list)

file_array = np.array(name_array)
print(file_array)

# CONVERT FILES USING LOOP
a = '.csv'
for x in range(44):
    data = pd.read_csv(new_array[x], on_bad_lines='skip')
    data.to_csv(file_array[x]+a, index=None)



