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
import biot_sav_sens as bass
import pyswarms as ps
import iris_data as iris
import my_izh_model as mim
import multiprocessing
from numba import jit

train_data = iris.give_train()

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
max_tau = 5
min_tau = 1
sim_length = 150
index = 60
num_of_NNs = 125
num_of_gens = 100
num_of_samps = 25

def load_object(filename):
	try:
		with open(filename, "rb") as f:
			return pickle.load(f)
	except Exception as ex:
		print("Error during unpickling object (Possibly unsupported):", ex)
 
mNN = load_object("myNeuralNetworkTemp.pickle")

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

def SetCurrent(NeuralNetwork, data_array):
	tempNeuralNetwork = copy.deepcopy(NeuralNetwork)
	of_interest = data_array[0]
	#print(data_array)
	for i in range(0, len(of_interest)):
		#tempNeuralNetwork[i].I = 10*of_interest[i]*on_off
		tempNeuralNetwork[i].I = 10*of_interest[i]
		#tempNeuralNetwork[i].I = 3.44
	return tempNeuralNetwork

def TimeStep(NeuralNetwork):
	tempNeuralNetwork = copy.deepcopy(NeuralNetwork)
	for n in range(0, num_of_all_neuron):
		#print(tempNeuralNetwork[n].I)
		tempNeuralNetwork[n] = mim.incr_volt(tempNeuralNetwork[n])

	for n in range(0, num_of_all_neuron):
		for m in range(0, len(tempNeuralNetwork[n].con_to)):
			i = tempNeuralNetwork[n].con_to[m]
			tempNeuralNetwork[i] = mim.incr_curr(tempNeuralNetwork[n], tempNeuralNetwork[i], m)	
	return(tempNeuralNetwork)

def DataToSpikes(TimeRange, data, NeuralNetwork):
	bufferArray = NeuralNetwork
	num_of_spikes = np.zeros(num_of_out_neuron)
	for t in range(0, TimeRange):
		bufferArray = SetCurrent(bufferArray, data)
		bufferArray = TimeStep(bufferArray)
		for j in range(0, num_of_out_neuron):
			num_of_spikes[j] += 1*bufferArray[num_of_all_neuron-num_of_out_neuron+j].justFired
	return num_of_spikes

def SpikesToIndex(ar):
	idk = []
	max_count = 0
	max_index = 4
	for i in range(0, len(ar)):
		idk.append(0)
		if ar[i] > max_count:
			max_count = ar[i]
			max_index = i
	if max_index != 4:
		idk[max_index] = 1
	#print(idk)
	return idk

def EvaluateNN(data, SpikesIndex):
	relevant_data = data[1]
	x = 3
	for i in range(0, len(relevant_data)):
		if relevant_data[i] == 1:
			x = i
	return x == SpikesIndex

def GenWeights():
	to_return_gbase = []
	to_return_tau = []
	for j in range(0, 5):
		to_return_gbase.append(random.uniform(min_gbase, max_gbase))
		to_return_tau.append(random.uniform(min_tau, max_tau))
	#print(to_return_gbase)
	return [to_return_gbase, to_return_tau]

def Mutate(NeuralNetwork):
	bufferNN = copy.deepcopy(NeuralNetwork)
	IndexToChange = random.randint(0, num_of_all_neuron-1-num_of_out_neuron) #picks a random neuron that isnt the output
	GbaseOrEX = random.random() #selects whether a gbase will change or whether the neuron is exitatory or inhibitory
	if GbaseOrEX > 0.4: #threshold for how frequent the types of changes are
		GbaseIndex = random.randint(0, 4)
		bufferNN[IndexToChange].gbases[GbaseIndex] = random.uniform(min_gbase, max_gbase)
	elif GbaseOrEX > 0.1:
		GbaseIndex = random.randint(0, 4)
		bufferNN[IndexToChange].taus[GbaseIndex] = random.uniform(min_tau, max_tau)
	else:
		exorin = random.random()
		if exorin >= 0.2:
			bufferNN[IndexToChange].exin = 1
		else:
			bufferNN[IndexToChange].exin = -1
	return bufferNN

def BigMutate(NeuralNetwork):
	bufferNN = copy.deepcopy(NeuralNetwork)
	for n in range(0, num_of_all_neuron-num_of_out_neuron):
		w = GenWeights()
		bufferNN[n].gbases = w[0]
		bufferNN[n].taus = w[1]
		exorin = random.random()
		if exorin >= 0.2:
			bufferNN[n].exin = 1
		else:
			bufferNN[n].exin = -1
	return (bufferNN)

def GenArrayOfNNs(how_many, ref_NN):
	toReturn = []
	for i in range(0, how_many):
		bufferNeuralNetwork = copy.deepcopy(ref_NN)
		toReturn.append(Mutate(bufferNeuralNetwork))
	return(toReturn)

def NumCorrect(NNofInterest=None, doi=train_data):
	num_correct = 0
	for t in range(0, num_of_samps):
		train = random.choice(doi)
		spikes = DataToSpikes(sim_length, train, NNofInterest)
		num_correct += 1*(SpikesToIndex(spikes) == train[1])
	return(num_correct)

def ParallelEval(ArrayOfNNs):
	pool = multiprocessing.Pool()
	pool = multiprocessing.Pool(processes=12)
	inputs = ArrayOfNNs
	outputs = pool.map(NumCorrect, inputs)
	return(outputs)

def save_object(obj):
	try:
		with open("GPTImprovedMIMIZH.pickle", "wb") as f:
			pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
	except Exception as ex:
		print("Error during pickling object (Possibly unsupported):", ex)

if __name__ == '__main__': #yeah idk why this is needed but multiprocessing just wont work without it
	MasterArrayOfNNs = GenArrayOfNNs(num_of_NNs, mNN.NeuralNetwork)
	def GenKeepIndex(ArrayOfNNs):
		results = ParallelEval(ArrayOfNNs)
		avg_score = np.mean(results)
		print(avg_score/num_of_samps, np.max(results)/num_of_samps)
		keep_index = []
		mu, sigma = 0, 0.01 # mean and standard deviation
		noise = np.random.normal(mu, sigma)
		for i in range(0, num_of_NNs):
			if results[i]+noise > avg_score:
				keep_index.append(i)
		return keep_index
	
	def GenNewPop(winners, ArrayOfNNs):
		tempNeuralNetworkArray = []
		for n in range(0, num_of_NNs):
			templateNNIndex = random.choice(winners)
			tempNeuralNetworkArray.append(Mutate(ArrayOfNNs[templateNNIndex]))
		return tempNeuralNetworkArray

	print("started loop")
	for i in range(0, num_of_gens):
		winners = GenKeepIndex(MasterArrayOfNNs)
		MasterArrayOfNNs = GenNewPop(winners, MasterArrayOfNNs)

	train_data = iris.give_test()
	num_of_samps = len(train_data)
	score_of_test = ParallelEval(MasterArrayOfNNs)
	max_score = 0
	save_index = 0
	for i in range(0, num_of_NNs):
		if score_of_test[i] > max_score:
			max_score = score_of_test[i]
			save_index = i
	print(max_score/num_of_samps)
	save_NN = myNetwork()
	save_NN.NeuralNetwork = MasterArrayOfNNs[i]
	save_object(save_NN)