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

for d in range(0, 150):
    if d%50 < 40:
        train_data.append(data_array[d])
    else:
        test_data.append(data_array[d])

def relu(x): #not relu anymore but hey
    if x < 0:
        return 0
    else:
        return x

def sigmoid(x):
    return np.arctan(x)


num_of_input = 4
num_of_hidden = 10
num_of_hidden_layers = 1
num_of_output = 3
template_input = np.random.rand(num_of_input)
template_output = np.random.rand(num_of_output)

mutation_weight = 0.05
threshold = 1.5
num_of_generation = 300
num_of_models = 500


class NN:
    def __init__(self): 
        self.input_layer = np.random.rand(num_of_input)
        self.hidden_layers = np.zeros((num_of_hidden_layers, num_of_hidden))
        self.output_layer = np.zeros(num_of_output)

        self.in_to_hid_wires = np.add(np.random.rand(num_of_hidden, num_of_input), np.ones((num_of_hidden, num_of_input))*(-0.5))
        self.hid_to_out_wires = np.add(np.random.rand(num_of_output, num_of_hidden), np.ones((num_of_output, num_of_hidden))*(-0.5))
        self.hid_to_hid_wires = np.add(np.random.rand(num_of_hidden_layers-1, num_of_hidden, num_of_hidden), np.ones((num_of_hidden_layers-1, num_of_hidden, num_of_hidden))*(-0.5))

def feed_forward(testNN):
    #takes in NN object and returns output array
    for h in range(0, num_of_hidden_layers):
        if h == 0:
            testNN.hidden_layers[h] =  np.matmul(testNN.in_to_hid_wires, testNN.input_layer)
            for n in range(0, num_of_hidden):
                testNN.hidden_layers[h][n] = sigmoid(testNN.hidden_layers[h][n])
        else:
            testNN.hidden_layers[h] = np.matmul(testNN.hid_to_hid_wires[h-1], testNN.hidden_layers[h-1])
            for n in range(0, num_of_hidden):
                testNN.hidden_layers[h][n] = sigmoid(testNN.hidden_layers[h][n])

    testNN.output_layer = np.matmul(testNN.hid_to_out_wires, testNN.hidden_layers[-1])
    for n in range(0, num_of_output):
        testNN.output_layer[n] = relu(testNN.output_layer[n])
    return(testNN.output_layer)





array_of_NNs = []
array_of_Scores = []
test_array_of_Scores = []



def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)
 
LoadedNN = load_object("avgNN.pickle")

def raw_to_max(ar):
    max_out = 0
    max_index = 0
    for i in range(0, len(ar)):
        if ar[i] > max_out:
            max_out = ar[i]
            max_index = i
    bleh = np.zeros(3)
    bleh[max_index] = 1
    return bleh

for test in test_data:
    LoadedNN.input_layer = test[0]

    print(test[1], raw_to_max(feed_forward(LoadedNN)))




