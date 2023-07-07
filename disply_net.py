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
import biot_sav_sens as bass
import pyswarms as ps
import iris_data as iris
import my_izh_model as mim

train_data = iris.give_train()

#ok new plan is to just gauss newton method this and use the currents to recreate weights as I assumed the current was a contant scalled by the synaptic weight
#probably need gpu accelerateration but who knows

class myNetwork():
	def __init__(self, NeuralNetwork=[]):
		self.NeuralNetwork = NeuralNetwork 

class myNeuron():

	def __init__(self, pos_x=0, pos_y=0, gbases=[], tim = 1111, con_to = [], a=0.02, b=0.2, c=-65, d=4, v=-65, u=-13, I=0, justFired = False, exin = 1): 
		self.pos_x = pos_x
		self.pos_y = pos_y
		self.gbases = gbases
		self.gbases = gbases
		self.con_to = con_to
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.v = v
		self.u = u
		self.I = I
		self.justFired = justFired
		self.exin = exin

mNN = myNetwork()
num_of_in_neuron = 4
num_of_hid_neuron = 40
num_of_out_neuron = 3
num_of_all_neuron = num_of_in_neuron+num_of_hid_neuron+num_of_out_neuron
train_data = iris.give_train()
max_gbase = 5
min_gbase = 0
sim_length = 200

def load_object(filename):
	try:
		with open(filename, "rb") as f:
			return pickle.load(f)
	except Exception as ex:
		print("Error during unpickling object (Possibly unsupported):", ex)
 
mNN = load_object("GPTImprovedMIMIZH.pickle")

def DisplayNeuralNetwork(NeuralNetwork):
	for n in range(0, len(NeuralNetwork)):
		Neur = NeuralNetwork[n]
		plt.annotate(n, xy = (Neur.pos_x, Neur.pos_y), ha='center', va='center')
		color  = "white"
		if Neur.exin == -1:
			color = [0, 0.7, .8]
		plt.plot(Neur.pos_x, Neur.pos_y, marker="o", markersize=15, markeredgecolor="black", markerfacecolor=color)
		
def DisplaySynapse(NeuralNetwork):
	for Neur in NeuralNetwork:
		#print(Neur.gbases)
		for i in range(0, len(Neur.con_to)):
			Xs = [Neur.pos_x, mNN.NeuralNetwork[Neur.con_to[i]].pos_x]
			Ys = [Neur.pos_y, mNN.NeuralNetwork[Neur.con_to[i]].pos_y]
			Color = [0, Neur.gbases[i]/(max_gbase), 0.2]
			plt.plot(Xs, Ys, color=Color)


DisplaySynapse(mNN.NeuralNetwork)
DisplayNeuralNetwork(mNN.NeuralNetwork)
plt.show()