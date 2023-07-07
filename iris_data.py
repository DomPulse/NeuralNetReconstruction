import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import random
import copy
import pickle
import csv

iris_file = "C:\\Users\\Richard\\Documents\\Neuro Sci\\Iris.csv"

data_array = []
label_array = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

with open(iris_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        new_row = row[0].split(",")
        #print(new_row)
        processed = []
        label = []
        for j in range(0, 3):
            label.append((new_row[5] == label_array[j])*1)
        #print(label)
        inputs = []
        for i in range(1, 5):
            inputs.append(float(new_row[i]))
        processed.append(inputs)
        processed.append(label)
        data_array.append(processed)

train_data = []
test_data = []

def normalize(ar):
    m0=7.9
    m1=4.4
    m2=6.9
    m3=2.5
    return [ar[0]/m0, ar[1]/m1, ar[2]/m2, ar[3]/m3]

def parse_data():
    for d in range(0, 150):
        data = data_array[d]
        data[0] = normalize(data[0])
        if d%50 < 40:
            train_data.append(data)
        else:
            test_data.append(data)


def give_train():

    return train_data

def give_test():
    
    return test_data

parse_data()












