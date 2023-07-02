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
import multiprocessing

class NN:
	def __init__(self): 
		self.input_layer = np.random.rand(num_of_input)
		self.hidden_layers = np.zeros((num_of_hidden_layers, num_of_hidden))
		self.output_layer = np.zeros(num_of_output)

		self.in_to_hid_wires = np.add(np.random.rand(num_of_hidden, num_of_input), np.ones((num_of_hidden, num_of_input))*(-0.5))
		self.hid_to_out_wires = np.add(np.random.rand(num_of_output, num_of_hidden), np.ones((num_of_output, num_of_hidden))*(-0.5))
		self.hid_to_hid_wires = np.add(np.random.rand(num_of_hidden_layers-1, num_of_hidden, num_of_hidden), np.ones((num_of_hidden_layers-1, num_of_hidden, num_of_hidden))*(-0.5))

def load_object(filename):
	try:
		with open(filename, "rb") as f:
			return pickle.load(f)
	except Exception as ex:
		print("Error during unpickling object (Possibly unsupported):", ex)
 
ReferenceNN = copy.deepcopy(load_object("IRIS.pickle"))
ReconNN = copy.deepcopy(load_object("4IRISRecon.pickle"))
for x in range(0, len(ReconNN.in_to_hid_wires)):
	for y in range(0, len(ReconNN.in_to_hid_wires[x])):
		print(ReferenceNN.in_to_hid_wires[x][y], ReconNN.in_to_hid_wires[x][y])
for x in range(0, len(ReconNN.hid_to_out_wires)):
	for y in range(0, len(ReconNN.hid_to_out_wires[x])):
		print("bleh")#ReferenceNN.hid_to_out_wires[x][y], ReconNN.hid_to_out_wires[x][y])
#print(ReferenceNN.in_to_hid_wires)
#print(ReconNN.in_to_hid_wires)
#print(ReferenceNN.hid_to_out_wires)
#print(ReconNN.hid_to_out_wires)



