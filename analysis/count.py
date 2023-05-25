import numpy as np
import pandas as pd
import os


files = "/{change this directory path to labeled txt data}/fixed_final/"
test_files_n = [files+'test/'+file for file in os.listdir(files+'test/') if ".txt" in file]
train_files_n = [files+'train/'+file for file in os.listdir(files+'train/') if ".txt" in file]
val_files_n   =[files+'valid/'+file for file in os.listdir(files+'valid/') if ".txt" in file]

test_files  = [np.loadtxt(file) for file in test_files_n]
train_files = [np.loadtxt(file) for file in train_files_n]
val_files   = [np.loadtxt(file) for file in val_files_n]

train = sum([len(file) for file in train_files])
val = (sum([len(file) for file in val_files]))
test = (sum([len(file) for file in test_files]))
print(train+val+test)
print(len(train_files), train, train/(train+val+test))
print(len(val_files), val, val/(train+val+test))
print(len(test_files), test, test/(train+val+test))

for i, train_files in enumerate(train_files):
    widths = train_files[:,3] * 3024
    heights = train_files[:,4] * 4032
    # largest circle inside the bounding box
    avg_radius = np.average(np.min([widths, heights], axis=0)/2)
    #print(avg_square_area_pixels)
    print('TRAIN ', i)
    print(train_files_n[i])
    print(avg_radius)
    print()
    
for i, val_files in enumerate(val_files):
    widths = val_files[:,3] * 3024
    heights = val_files[:,4] * 4032
    # largest circle inside the bounding box
    avg_radius = np.average(np.min([widths, heights], axis=0)/2)
    #print(avg_square_area_pixels)
    print('VAL ', i)
    print(val_files_n[i])
    print(avg_radius)
    print()


for i, test_files in enumerate(test_files):
    widths = test_files[:,3] * 3024
    heights = test_files[:,4] * 4032
    # largest circle inside the bounding box
    avg_radius = np.average(np.min([widths, heights], axis=0)/2)
    #print(avg_square_area_pixels)
    print('TEST ', i)
    print(test_files_n[i])
    print(avg_radius)
    print()