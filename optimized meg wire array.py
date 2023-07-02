import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import random
import copy
import pickle
import matplotlib.pyplot as plt
import csv
import time

#defines paramaters of the sensor array and initializes values
num_of_row = 4
num_of_col = num_of_row
sensor_height = 0.01
SensorValues = np.zeros((num_of_row, num_of_col))

#paramaters of the particular neural netork being tested, will have to make this more general later
num_of_input = 4
num_of_hidden = 10
num_of_hidden_layers = 1
num_of_output = 3

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

class NN:
    def __init__(self): 
        self.input_layer = np.random.rand(num_of_input)
        self.hidden_layers = np.zeros((num_of_hidden_layers, num_of_hidden))
        self.output_layer = np.zeros(num_of_output)

        self.in_to_hid_wires = np.add(np.random.rand(num_of_hidden, num_of_input), np.ones((num_of_hidden, num_of_input))*(-0.5))
        self.hid_to_out_wires = np.add(np.random.rand(num_of_output, num_of_hidden), np.ones((num_of_output, num_of_hidden))*(-0.5))
        self.hid_to_hid_wires = np.add(np.random.rand(num_of_hidden_layers-1, num_of_hidden, num_of_hidden), np.ones((num_of_hidden_layers-1, num_of_hidden, num_of_hidden))*(-0.5))

class Sensor:
    def __init__(self): 
    	self.position = np.zeros(2) #position on xy plane

    	self.value = 0 #the overall value of each sensor from a collection of wires with a particular current value

    	self.in_to_hid_wires_contribute = np.zeros((num_of_hidden, num_of_input)) #contribution from each "wire" in each layer
    	self.hid_to_out_wires_contribute = np.zeros((num_of_output, num_of_hidden))
    	self.hid_to_hid_wires_contribute = np.zeros((num_of_hidden_layers-1, num_of_hidden, num_of_hidden))

    	self.in_to_hid_wires_current = np.zeros((num_of_hidden, num_of_input)) #"current" of each wire in each layer, will act as a multiplier toward that wires contribution
    	self.hid_to_out_wires_current = np.zeros((num_of_output, num_of_hidden))
    	self.hid_to_hid_wires_current = np.zeros((num_of_hidden_layers-1, num_of_hidden, num_of_hidden))

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
    return(testNN)

def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)
 
LoadedNN = load_object("uuuuuuuummmmmmm.pickle")
LoadedNN.input_layer = test_data[1][0]
print(LoadedNN.input_layer)
LoadedNN = feed_forward(LoadedNN)
print(LoadedNN.output_layer)

def clusters_of_wires(num_of_in_neurons, num_of_out_neurons, start_x_pos, end_x_pos, sens_x, sens_y):
	wire_cluster_cont_array = np.zeros((num_of_out_neurons, num_of_in_neurons))
	for cluster in range(0, num_of_in_neurons):
		for wire in range(0, num_of_out_neurons):
			x0 = start_x_pos
			xf = end_x_pos
			y0 = (cluster+1)/(num_of_in_neurons+1)
			yf = (wire+1)/(num_of_out_neurons+1)
			wire_cluster_cont_array[wire][cluster] = get_sensor_val_contribute(x0, y0, xf, yf, 1, sens_x, sens_y)
	return wire_cluster_cont_array 

def bio_savt_step(dl, not_r): #all physical constants ignored like a chad
	r_hat = np.multiply(not_r, 1/np.sqrt(not_r[0]**2+not_r[1]**2+not_r[2]**2))
	#print(np.sqrt(r_hat[0]**2+r_hat[1]**2+r_hat[2]**2))
	B_vect = np.cross(dl, r_hat)
	r_mag = np.sqrt(not_r[0]**2+not_r[1]**2+not_r[2]**2)
	#print(np.sqrt(not_r[0]**2+not_r[1]**2+not_r[2]**2))
	B_vect = np.multiply(B_vect, 1/(r_mag*r_mag))
	return B_vect

def get_sensor_val_contribute(wire_x0, wire_y0, wire_xf, wire_yf, wire_current, sens_x, sens_y):

	totalB = [0, 0, 0]
	#totalB = 0
	m = (wire_yf-wire_y0)/(wire_xf-wire_x0)
	x = wire_x0
	dx = 0.01 #arbitrary, sets resolution
	while x < wire_xf:

		y = (x-wire_x0)*m+wire_y0

		r = [sens_x-x, sens_y-y, sensor_height]
		dl = [dx, m*dx, 0]
		totalB = np.add(bio_savt_step(dl, r), totalB)
		x += dx
	return np.multiply(totalB, wire_current)[2]

def gen_sensor_array(num_of_col, num_of_row):
	TempSensorArray = [] #creates an array to contain all sensor objects
	count = 0
	for x in range(0, num_of_col):
		for y in range(0, num_of_row):
			TempSensorArray.append(Sensor())
			x_pos = x/(num_of_row-1)
			y_pos = y/(num_of_col-1)
			TempSensorArray[count].position = [x_pos, y_pos]
			x_offset = 0
			spacing = 1/(num_of_hidden_layers+1) #this might be wrong but boy will it work with this particular NN
			for layer in range(0, num_of_hidden_layers+1):
				if layer == 0:
					TempSensorArray[count].in_to_hid_wires_contribute = clusters_of_wires(num_of_input, num_of_hidden, x_offset, x_offset+spacing, x_pos, y_pos)
				elif layer == num_of_hidden_layers:
					TempSensorArray[count].hid_to_out_wires_contribute = clusters_of_wires(num_of_hidden, num_of_output, x_offset, x_offset+spacing, x_pos, y_pos)
				else:
					TempSensorArray[count].hid_to_hid_wires_contribute[layer-1] = clusters_of_wires(num_of_hidden, num_of_hidden, x_offset, x_offset+spacing, x_pos, y_pos)
				x_offset += spacing 
				#print(x_offset)
			count += 1
	return TempSensorArray

def get_sensors_from_NN_state(SensorArray, NN_State):
	#need to properly impliment multiple hidden layer
	#test case with the iris NN only has one so I'm not bothering 
	#since i dont have an easy debug case id probably just mess up anyway
	NewSensorArray = []
	index = 0
	for Sensor in SensorArray:
		Sensor.value = 0
		for InNode in range(0, len(NN_State.input_layer)):
			for OutNode in range(0, len(NN_State.hidden_layers[0])):
				Sensor.value += NN_State.input_layer[InNode]*NN_State.in_to_hid_wires[OutNode][InNode]*Sensor.in_to_hid_wires_contribute[OutNode][InNode]
		for InNode in range(0, len(NN_State.hidden_layers[0])):
			for OutNode in range(0, len(NN_State.output_layer)):
				Sensor.value += NN_State.hidden_layers[0][InNode]*NN_State.hid_to_out_wires[OutNode][InNode]*Sensor.hid_to_out_wires_contribute[OutNode][InNode]
		NewSensorArray.append(Sensor)
		print(NewSensorArray[index].value)
		index += 1
	return NewSensorArray	

def display_sensor_array(SensorArray):
	for Sensor in SensorArray:
		mult = np.arctan(Sensor.value*0.1)/(np.pi/2)
		if mult >= 0:
			Color = [1*mult, 0, 0]
		else:
			Color = [0, -1*mult, 0]
		plt.plot(Sensor.position[0], Sensor.position[1], marker="o", markersize=15, markeredgecolor="white", markerfacecolor=Color)

SensorArray = gen_sensor_array(num_of_col, num_of_row)
SensorArray = get_sensors_from_NN_state(SensorArray, LoadedNN)
display_sensor_array(SensorArray)

plt.savefig('bleh.jpg',bbox_inches='tight', dpi=1000)
plt.show()