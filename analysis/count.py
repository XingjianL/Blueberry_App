import numpy as np
import pandas as pd
import os

#files = "/home/lixin/Classes/Fall22Lab/annotated_blueberry/yolo_dataset/fixed/"
files = "/home/lixin/Classes/Fall22Lab/annotated_blueberry/yolo_dataset/fixed_final/"
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

for i, trainfile in enumerate(train_files):
    widths = trainfile[:,3] * 3024
    heights = trainfile[:,4] * 4032
    avg_square_area_pixels = np.average(widths * heights)
    #print(avg_square_area_pixels)
    print(train_files_n[i])
    print(np.sqrt(avg_square_area_pixels/np.pi))
    print()