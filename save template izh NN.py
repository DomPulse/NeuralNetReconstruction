import numpy as np
from os.path  import join
import random
import copy
import pickle
import matplotlib.pyplot as plt
import biot_sav_sens as bass
import iris_data as iris
import my_izh_model as mim

seed = 42
random.seed(seed)

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
num_of_synapse = 0
train_data = iris.give_train()
max_gbase = 5
min_gbase = 0
max_tau = 5
min_tau = 1
perc_of_exorin = 0.2 #this sets the percentage of inhibitory neurons, probabalistic

#make a list of all possible x y cooridinates, randomly pick on from it and then remove it from the list
allowed_space = []
for x in range(1, 9):
	for y in range(0, 10):
		allowed_space.append([x, y])
virgin_space = allowed_space
taken_space = []
taken_space_dict = {}

def GenWeights():
	to_return_gbase = []
	to_return_tau = []
	for j in range(0, 5):
		to_return_gbase.append(random.uniform(min_gbase, max_gbase))
		to_return_tau.append(random.uniform(min_tau, max_tau))
	#print(to_return_gbase)
	return [to_return_gbase, to_return_tau]

def AddNeuron(noin, nohn, noon):
	tempNeurArray = []
	for i in range(0, noin):
		pos = [0, int(9*i/(noin-1))]
		w = GenWeights()
		exorin = ((random.random()>perc_of_exorin)*2)-1
		print(exorin)
		tempNeurArray.append(myNeuron(pos[0], pos[1], w[1], w[0], exin = exorin))
		taken_space.append(pos)
		taken_space_dict[str(pos)] = i
	for i in range(0, nohn):
		exorin = ((random.random()>perc_of_exorin)*2)-1
		pos_index = random.randrange(0, len(allowed_space))
		pos = allowed_space[pos_index]
		taken_space.append(pos)
		taken_space_dict[str(pos)] = i+noin
		w = GenWeights()
		tempNeurArray.append(myNeuron(pos[0], pos[1], w[1], w[0], exin = exorin))
		del allowed_space[pos_index]
	for i in range(0, noon):
		exorin = ((random.random()>perc_of_exorin)*2)-1
		pos = [9, int(9*i/(noon-1))]
		w = GenWeights()
		tempNeurArray.append(myNeuron(pos[0], pos[1], w[1], w[0], exin = exorin))
		taken_space.append(pos)
		taken_space_dict[str(pos)] = i+noin+nohn
	return tempNeurArray

def GrowSyn(taken_space, Neur):
	#print(taken_space)
	pre_x = Neur.pos_x
	pre_y = Neur.pos_y
	post_x = pre_x
	post_ys = np.linspace(pre_y-2, pre_y+2, 5)
	#print(pre_y, post_ys)
	con_to_ind = []
	while post_x < int(pre_x+2):
		post_x += 1
		post_x = int(post_x)
		to_del = []
		for i in range(0, len(post_ys)):
			post_y = int(post_ys[i])
			#print([post_x, post_y])
			if [post_x, post_y] in taken_space:
				#print("hit")
				#plt.plot([pre_x, post_x], [pre_y, post_y], "black")
				to_del.append(i)
				con_to_ind.append(taken_space_dict[str([post_x, post_y])])
				
		np.delete(post_ys, to_del)
		copy_ys = []
		for j in range(0, len(post_ys)):
			if j not in to_del:
				copy_ys.append(post_ys[j])
		post_ys = copy_ys
	return con_to_ind
				
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

mNN.NeuralNetwork = AddNeuron(num_of_in_neuron, num_of_hid_neuron, num_of_out_neuron)

for n in range(0, num_of_all_neuron):
	#print(mNN.NeuralNetwork[n].con_to)
	mNN.NeuralNetwork[n].con_to = GrowSyn(taken_space, mNN.NeuralNetwork[n])
	num_of_synapse += len(mNN.NeuralNetwork[n].con_to)

DisplaySynapse(mNN.NeuralNetwork)
DisplayNeuralNetwork(mNN.NeuralNetwork)

print(num_of_synapse)
print(num_of_all_neuron)


def save_object(obj):
    try:
        with open("myNeuralNetworkTemp.pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
save_object(mNN)
plt.show()