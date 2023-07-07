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
import gen_mag_sens_matrix as gmsm
import iris_data as iris
import my_izh_model as mim

train_data = iris.give_train()

#ok new plan is to just gauss newton method this and use the currents to recreate weights as I assumed the current was a contant scalled by the synaptic weight
#probably need gpu accelerateration but who knows

class myNetwork():
	def __init__(self, NeuralNetwork=[]):
		self.NeuralNetwork = NeuralNetwork 

class myNeuron():

	def __init__(self, pos_x=0, pos_y=0, taus=[], gbases=[], tim = 1111, con_to = [], a=0.02, b=0.2, c=-65, d=4, v=-65, u=-13, I=0, justFired = False, exin = 1): 
		self.pos_x = pos_x
		self.pos_y = pos_y
		self.taus = taus
		self.gbases = gbases
		self.tim = tim
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
	for Neur in NeuralNetwork:
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

def SetCurrent(NeurArray, data_array):
	tempNeurArray = copy.deepcopy(NeurArray)
	of_interest = data_array[0]
	#print(data_array)
	for i in range(0, len(of_interest)):
		#tempNeurArray[i].I = 10*of_interest[i]*on_off
		tempNeurArray[i].I = 10*of_interest[i]
		#tempNeurArray[i].I = 3.44
	return tempNeurArray

def TimeStep(NeurArray):
	tempNeurArray = copy.deepcopy(NeurArray)
	for n in range(0, num_of_all_neuron):
		#print(tempNeurArray[n].I)
		tempNeurArray[n] = mim.incr_volt(tempNeurArray[n])

	for n in range(0, num_of_all_neuron):
		for m in range(0, len(tempNeurArray[n].con_to)):
			i = tempNeurArray[n].con_to[m]
			tempNeurArray[i] = mim.incr_curr(tempNeurArray[n], tempNeurArray[i], m)	
	return(tempNeurArray)

def DataToSpikes(TimeRange, data, NeuralNetwork):
	bufferArray = NeuralNetwork
	num_of_spikes = np.zeros(num_of_out_neuron)
	for t in range(0, TimeRange):

		bufferArray = SetCurrent(bufferArray, data)
		bufferArray = TimeStep(bufferArray)
		print(FindMagFieldAtTime(bufferArray))
		for j in range(0, num_of_out_neuron):
			num_of_spikes[j] += 1*bufferArray[num_of_all_neuron-num_of_out_neuron+j].justFired

	return num_of_spikes

def SpikesToIndex(ar):
	max_count = ar[0]
	max_index = 0
	for i in range(1, len(ar)):
		if ar[i] > max_count:
			max_count = ar[i]
			max_index = i
	return max_index

def FindMagFieldAtTime(NeuralNetwork):
	Currents = []
	for n in range(0, num_of_all_neuron):
		pre_n = NeuralNetwork[n]
		for l in range(0, len(pre_n.con_to)):
			post_index = pre_n.con_to[l]
			buffer_neuron = NeuralNetwork[post_index]
			#this current code is identical to that in my_izh_model, should be a function but whatever
			if pre_n.exin == 1:
				current = 5*pre_n.gbases[l]*(-buffer_neuron.v)*mim.Conducks(pre_n.tim, pre_n.taus[l])
			else:
				current += 5*pre_n.gbases[l]*(-buffer_neuron.v-65)*mim.Conducks(pre_n.tim, pre_n.taus[l])
			Currents.append(current)
	Currents = np.array(Currents)
	return np.matmul(Currents, ConvertMat)

index = 119
ConvertMat = gmsm.GenConvertMat(mNN.NeuralNetwork)
bleh = DataToSpikes(sim_length, train_data[index], mNN.NeuralNetwork)
print(bleh)
print(train_data[index])

plt.show()