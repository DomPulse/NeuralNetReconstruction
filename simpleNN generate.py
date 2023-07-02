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

def invtan(x):
	
	return np.arctan(x)

class NN:
	def __init__(self): 
		self.input_layer = np.random.rand(num_of_input)
		self.hidden_layers = np.zeros((num_of_hidden_layers, num_of_hidden))
		self.output_layer = np.zeros(num_of_output)

		self.in_to_hid_wires = np.add(np.random.rand(num_of_hidden, num_of_input), np.ones((num_of_hidden, num_of_input))*(-0.5))
		self.hid_to_out_wires = np.add(np.random.rand(num_of_output, num_of_hidden), np.ones((num_of_output, num_of_hidden))*(-0.5))
		self.hid_to_hid_wires = np.add(np.random.rand(num_of_hidden_layers-1, num_of_hidden, num_of_hidden), np.ones((num_of_hidden_layers-1, num_of_hidden, num_of_hidden))*(-0.5))


def better_clusters_of_wires(num_of_in_neurons, num_of_out_neurons, start_x_pos, end_x_pos, sens_x, sens_y):
	wire_cluster_cont_array = np.zeros((num_of_in_neurons, num_of_out_neurons))
	for cluster in range(0, num_of_in_neurons):
		for wire in range(0, num_of_out_neurons):
			x0 = start_x_pos
			xf = end_x_pos
			y0 = (cluster+1)/(num_of_in_neurons+1)
			yf = (wire+1)/(num_of_out_neurons+1)
			wire_cluster_cont_array[cluster][wire] = get_sensor_val_contribute(x0, y0, xf, yf, 1, sens_x, sens_y)
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

def get_wire_currents(NN_State, num_of_wires):
	TempCurrentMatrix = np.random.rand(1, num_of_wires)
	temp_index = 0
	for InNode in range(0, len(NN_State.input_layer)):
		for OutNode in range(0, len(NN_State.hidden_layers[0])):
			TempCurrentMatrix[0][temp_index] = NN_State.input_layer[InNode]*NN_State.in_to_hid_wires[OutNode][InNode]
			temp_index += 1
	for InNode in range(0, len(NN_State.hidden_layers[0])):
		for OutNode in range(0, len(NN_State.output_layer)):
			TempCurrentMatrix[0][temp_index] = NN_State.hidden_layers[0][InNode]*NN_State.hid_to_out_wires[OutNode][InNode]
			temp_index += 1
	return TempCurrentMatrix


def gen_conversion_matrix(num_of_wires, num_of_col, num_of_row):
	TempConversionMatrix = np.random.rand(num_of_wires, num_of_col*num_of_row)
	TempSensorArray = [] #creates an array to contain all sensor objects

	count = 0
	sensor_index = 0
	wire_index = 0
	for x in range(0, num_of_col):
		for y in range(0, num_of_row):
			x_pos = x/(num_of_row-1)
			y_pos = y/(num_of_col-1)
			x_offset = 0
			spacing = 1/(num_of_hidden_layers+1) #this might be wrong but boy will it work with this particular NN
			wire_index = 0
			for layer in range(0, num_of_hidden_layers+1):
				if layer == 0:
					cluster_of_interest = better_clusters_of_wires(num_of_input, num_of_hidden, x_offset, x_offset+spacing, x_pos, y_pos)
					for input_node in cluster_of_interest:
						for wire in input_node:
							TempConversionMatrix[wire_index][sensor_index] = wire
							wire_index += 1
				elif layer != num_of_hidden_layers:
					cluster_of_interest = better_clusters_of_wires(num_of_hidden, num_of_hidden, x_offset, x_offset+spacing, x_pos, y_pos)

					for input_node in cluster_of_interest:
						for wire in input_node:
							TempConversionMatrix[wire_index][sensor_index] = wire
							wire_index += 1
				else:
					cluster_of_interest = better_clusters_of_wires(num_of_hidden, num_of_output, x_offset, x_offset+spacing, x_pos, y_pos)

					for input_node in cluster_of_interest:
						for wire in input_node:
							TempConversionMatrix[wire_index][sensor_index] = wire
							wire_index += 1
				x_offset += spacing 
				#print(x_offset)
			count += 1
			sensor_index += 1
	return TempConversionMatrix

def feed_forward(testNN): #this feed forward has been modified to exclued relu, dont just use it
	#takes in NN object and returns output array
	for h in range(0, len(testNN.hidden_layers)):
		if h == 0:
			testNN.hidden_layers[h] =  np.matmul(testNN.in_to_hid_wires, testNN.input_layer)
			for n in range(0, len(testNN.hidden_layers[0])):
				testNN.hidden_layers[h][n] = invtan(testNN.hidden_layers[h][n])
		else:
			testNN.hidden_layers[h] = np.matmul(testNN.hid_to_hid_wires[h-1], testNN.hidden_layers[h-1])
			for n in range(0, len(testNN.hidden_layers[0])):
				testNN.hidden_layers[h][n] = invtan(testNN.hidden_layers[h][n])

	testNN.output_layer = np.matmul(testNN.hid_to_out_wires, testNN.hidden_layers[-1])

	return(testNN)

num_of_input = 2
num_of_hidden = 2
num_of_hidden_layers = 1
num_of_output = 1
num_of_col = 2
num_of_row = num_of_col
sensor_height = 0.01
SimpNN = NN()

w = len(SimpNN.input_layer)*len(SimpNN.hidden_layers[0]) + len(SimpNN.hidden_layers[0])*len(SimpNN.hidden_layers[0])*(len(SimpNN.hidden_layers)-1) +  len(SimpNN.output_layer)*len(SimpNN.hidden_layers[0])
s = num_of_col*num_of_row

SimpNN.input_layer = np.ones(2)
SimpNN = feed_forward(SimpNN)
print(SimpNN.input_layer, SimpNN.output_layer)

for i in range(0, 5):
	SimpNN.input_layer = np.random.rand((2))
	SimpNN = feed_forward(SimpNN)
	print(SimpNN.input_layer, SimpNN.output_layer)
	CurrentMatrix = get_wire_currents(SimpNN, w)
	ConversionMatrix = gen_conversion_matrix(w, num_of_col, num_of_row)
	SensorMatrix = np.matmul(CurrentMatrix, ConversionMatrix)
	print(SensorMatrix)

SimpNN.input_layer = np.ones(2)#this check is just here to make sure i'm not overwriting the NN somehow or something goofy
SimpNN = feed_forward(SimpNN)
print(SimpNN.input_layer, SimpNN.output_layer)

def save_object(obj):
	try:
		with open("EvolutionViaCurrentsNotRecon.pickle", "wb") as f:
			pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
	except Exception as ex:
		print("Error during pickling object (Possibly unsupported):", ex)
 
save_object(SimpNN)