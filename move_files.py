import os
import re
import shutil

"""
Move any csv files that do not have a folder into a folder
"""
#Locate folder containing csv files
folder_path = os.path.join(os.getcwd(),'2023_2024')
files = os.listdir(folder_path)

#patterns to extract from filenames
pattern = re.compile(r'experiment_data_pID_(.*?)_')
pattern2 = re.compile(r'convergence_distance_pID_(.*?)_')

file_dict = {}

for file in files:
    #check for csv files
    if file.endswith(".csv"):
        #extract pattern 1
        match = pattern.search(file)
        match2 = pattern2.search(file)

        if match:
            extracted_string = match.group(1)
            #create folder if it does not exist
            if extracted_string not in file_dict:
                os.makedirs(os.path.join(folder_path, extracted_string), exist_ok=True)
                file_dict[extracted_string] = []
            #add the filename
            file_dict[extracted_string].append(file)

        if match2:
            extracted_string = match2.group(1)
            #create folder if it does not exist
            if extracted_string not in file_dict:
                os.makedirs(os.path.join(folder_path, extracted_string), exist_ok=True)
                file_dict[extracted_string] = []
            #add the filename
            file_dict[extracted_string].append(file)

#move csv files
for key, value in file_dict.items():
    folder_name = os.path.join(folder_path, key)
    for file in value:
        source = os.path.join(folder_path, file)
        destination = os.path.join(folder_name, file)
        shutil.move(source, destination)