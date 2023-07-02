#ok big think to change is just have a better way of indexing variables/wire, its a huge headache
#add thermal noise to sensor values
#add bounds to the weights

import math
import numpy as np
import random
from sympy import *
import matplotlib.pyplot as plt
import copy
import struct
from array import array
from os.path  import join
import random
import pickle
import csv
import multiprocessing
import time


num_of_input = 4
num_of_hidden = 10
num_of_hidden_layers = 1
num_of_output = 3
num_of_col = 8 #try 4x3
num_of_row = 8
sensor_height = 0.01
num_of_thots = 5 #try 50
alpha = 0.5
noise_mag = 0.005

def other_save_object(obj, name):
		try:
			with open(name, "wb") as f:
				pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
		except Exception as ex:
			print("Error during pickling object (Possibly unsupported):", ex)

def find_alpha(d_array, ex_in_state, ex_sens_state, c_array, w_array):
	win_diff_mag = 10000
	win_alpha = 1
	for alph in range(0, 10):
		alpha  = (alph+1)/10
		test_wires = w_array + np.multiply(alpha, d_array)
		test_NN = weights_to_NN(test_wires)
		test_NN.input_layer = ex_in_state[0]
		test_NN = feed_forward(test_NN)
		CurrentMatrix = get_wire_currents(test_NN, w)
		SensorMatrix = np.matmul(CurrentMatrix, c_array)
		diff = SensorMatrix - ex_sens_state[0]
		diff_mag = magnitude(diff[0])
		if diff_mag < win_diff_mag:
			win_diff_mag = diff_mag
			win_alpha = alpha

	return np.multiply(win_alpha, d_array)

def bound_vars(weight):
	if weight > 2:
		return 2
	if weight < -2:
		return -2
	return weight

def gen_math_mat(nop, nov, f_array, v_array, v_guess):
	subs_array = []
	for i in range(0, len(v_array)):
		subs_array.append((v_array[i], v_guess[i]))
	Jacob = []
	for i in range(0, nop):
		to_append = []
		for j in range(0, nov):
			deriv = diff(f_array[i], v_array[j])
			
			deriv = deriv.subs(subs_array)
			to_append.append(deriv)
		Jacob.append(to_append)
	Jacob = Matrix(Jacob)
	JacobT = Jacob.transpose()
	JacJacT = JacobT*Jacob
	#print((JacJacT))
	InvJacJacT = JacJacT.inv() #idk this is weird for some reason
	FinalMat = InvJacJacT*JacobT
	return FinalMat

def iterate(arg_array):
	var_array = arg_array[0]
	var_guess = arg_array[1]
	workingNN = arg_array[2]
	s_state_array = arg_array[3]
	in_state_array = arg_array[4]
	c_array = arg_array[5]
	n_mag = arg_array[6]
	mag = 1000
	iters = 1
	var_n_mag = n_mag
	while mag > 0.2:

		subs_array = []
		for i in range(0, len(var_array)):
			subs_array.append((var_array[i], var_guess[i]))
		
		#gaoing to have to generate the function array and the math_mat from scratch each iteration to save myself from a bunch of symbolic math which might be even slower
		func_array = []
		for i in range(0, len(in_state_array)):
			workingNN.input_layer = in_state_array[i]
			workingNN = feed_forward(workingNN)
			temp_func_array = gen_funcs(var_array, s_state_array[i], workingNN, c_array, var_n_mag) #yeah this needs to be reworked to include the whole arctand erivative thing
			for func in temp_func_array:
				func_array.append(func)
		m_mat = gen_math_mat(len(func_array), len(var_array), func_array, var_array, var_guess)
		for f in range(0, len(func_array)):
			
			func_array[f] = func_array[f].subs(subs_array) #this doesn't quite work
		#print(temp_func_array)
		func_array = Matrix(func_array)
		delta = m_mat*func_array

		del_array = []
		for d in delta:
			del_array.append(alpha*d)
		var_guess = var_guess - del_array
		for var in range(0, len(var_guess)):
			var_guess[var] = bound_vars(var_guess[var])
		workingNN = weights_to_NN(var_guess)
		workingNN.input_layer = np.ones(num_of_input) #this is hard coded, function should be more flexible in terms of inputs and in terms of gathering information from multiple inputs to the og NN
		workingNN = feed_forward(workingNN)

		mag = magnitude(del_array)
		print(iters, mag, workingNN.output_layer, var_n_mag, str(arg_array[7]), flush = True)
		iters += 1
		var_n_mag -= 0.00025
		if var_n_mag < 0:
			var_n_mag = 0
		if iters%35 == 0:
			var_n_mag += 0.0025
		ReconNN = weights_to_NN(var_guess)
		other_save_object(ReconNN, str(arg_array[7])+"IRISRecon.pickle")

	
	return var_guess

def invtan(x):
	
	return np.arctan(x)

def magnitude(vector):

    return math.sqrt(sum(pow(element, 2) for element in vector))

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

def weights_to_NN(wire_array):
	tempNN = NN()
	temp_index = 0
	for InNode in range(0, len(tempNN.input_layer)):
		for OutNode in range(0, len(tempNN.hidden_layers[0])):
			tempNN.in_to_hid_wires[OutNode][InNode] = wire_array[temp_index]
			temp_index += 1
	for InNode in range(0, len(tempNN.hidden_layers[0])):
		for OutNode in range(0, len(tempNN.output_layer)):
			tempNN.hid_to_out_wires[OutNode][InNode] = wire_array[temp_index]
			temp_index += 1
	return tempNN

def gen_funcs(v_array, s_array, testNN, c_array, n_mag):
	func_array = []
	for f in range(0, len(s_array[0])):
		temp_func = -s_array[0][f] + (random.random()-0.5)*2*noise_mag*s_array[0][f]
		temp_index = 0
		for InNode in range(0, len(testNN.input_layer)):
			for OutNode in range(0, len(testNN.hidden_layers[0])):
				temp_func = temp_func + Mul(v_array[temp_index], c_array[temp_index][f]*testNN.input_layer[InNode])
				temp_index += 1
		for InNode in range(0, len(testNN.hidden_layers[0])):
			for OutNode in range(0, len(testNN.output_layer)):
				PrevIndex = InNode #need a way to get the wires from a particular HIDDEN node
				arg = 0
				for PrevNode in range(0, len(testNN.input_layer)):
					arg = arg + Mul(v_array[PrevIndex], testNN.input_layer[PrevNode]) #this is wrong, need a smarter indexing method
					PrevIndex += len(testNN.hidden_layers[0])
				temp_func = temp_func + Mul(v_array[temp_index], c_array[temp_index][f]*atan(arg))
				temp_index += 1
		#print(temp_func)
		func_array.append(temp_func)
	#print(func_array)
	return func_array

def load_object(filename):
	try:
		with open(filename, "rb") as f:
			return pickle.load(f)
	except Exception as ex:
		print("Error during unpickling object (Possibly unsupported):", ex)



if __name__ == '__main__':
	SimpNN = NN()
	SimpNN = load_object("IRIS.pickle")

	w = len(SimpNN.input_layer)*len(SimpNN.hidden_layers[0]) + len(SimpNN.hidden_layers[0])*len(SimpNN.hidden_layers[0])*(len(SimpNN.hidden_layers)-1) +  len(SimpNN.output_layer)*len(SimpNN.hidden_layers[0])
	s = num_of_col*num_of_row


	weight_approx_array = []
	for x in range(0, 5):
		weight_approx_array.append(np.random.rand(w)-0.5)
	#gen functions
	#gen variables
	#gonna have to generate new functions and matricies each time bc if I want them to be linear it's gonna take in the hidden layer values if that makes sense
	var_array = []
	for x in range(0, w):

		var_array.append(Symbol("w"+str(x))) #these are the weights of the neural network

	SimpNN.input_layer = np.ones(num_of_input)
	SimpNN = feed_forward(SimpNN)
	print(SimpNN.input_layer, SimpNN.output_layer)
	CurrentMatrix = get_wire_currents(SimpNN, w)
	ConversionMatrix = gen_conversion_matrix(w, num_of_col, num_of_row)
	SensorMatrix = np.matmul(CurrentMatrix, ConversionMatrix)

	input_states = []
	sensor_mat_states = []
	for state in range(0, num_of_thots):
		input_states.append(np.random.rand(num_of_input))
		SimpNN.input_layer = input_states[state]
		SimpNN = feed_forward(SimpNN)
		CurrentMatrix = get_wire_currents(SimpNN, w)
		SensorMatrix = np.matmul(CurrentMatrix, ConversionMatrix)
		sensor_mat_states.append(SensorMatrix)

	def save_object(obj, name):
		try:
			with open(name, "wb") as f:
				pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
		except Exception as ex:
			print("Error during pickling object (Possibly unsupported):", ex)
	pool = multiprocessing.Pool()
	pool = multiprocessing.Pool(processes=5)
	inputs = []
	for l in range(0, len(weight_approx_array)):
		to_append = []
		to_append.append(var_array)
		to_append.append(weight_approx_array[l])
		to_append.append(weights_to_NN(weight_approx_array[l]))
		to_append.append(sensor_mat_states)
		to_append.append(input_states)
		to_append.append(ConversionMatrix)
		to_append.append(noise_mag)
		to_append.append(l)
		inputs.append(to_append)


	
	outputs = pool.map(iterate, inputs)
	for l in range(0, len(outputs)):
		ReconNN = weights_to_NN(outputs[l])
		save_object(ReconNN, str(l)+"IRISRecon.pickle")

 



