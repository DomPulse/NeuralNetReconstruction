import numpy as np
import iris_data as iris
import matplotlib.pyplot as plt



curr_mult = 10
exin_array = [1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1]
exin_array = [1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1]

class myNeuron():
	def __init__(self, V=-65, u=-13, a=0.02, b=0.2, c=-65, d=4,  I=0, jF = False, exin = 1): 
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.V = V
		self.u = u
		self.I = I
		self.jF = jF
		self.exin = exin

num_neurons = 100
num_models = 50
num_samps = 20
num_gens = 50
num_mutate = 10
max_volt = 1
sim_length = 1000

def dif_v(v, u, I):

	return (0.04*v**2)+(5*v)+140-u+I

def dif_u(v, u, a, b):

	return a*(b*v-u)

def step_v(NeurArray):
	for NeurIdx in range(0, len(NeurArray)):
		NeurObj = NeurArray[NeurIdx]
		NeurObj.V += dif_v(NeurObj.V, NeurObj.u, NeurObj.I)
		NeurObj.u += dif_u(NeurObj.V, NeurObj.u, NeurObj.a, NeurObj.b)
		if NeurObj.jF == True:
			NeurObj.jF = False
			NeurObj.V = NeurObj.c
			NeurObj.u = NeurObj.u + NeurObj.d
		
		if NeurObj.V >= -30:
			NeurObj.V = 30
			NeurObj.jF = True
		

		if NeurObj.V <= -70:
			NeurObj.V = -70
		

		NeurArray[NeurIdx] = NeurObj
	
	return NeurArray

def step_I(NeurArray, SynArray):
	#this formerly acted on the current input I 
	#has since been changed to act on voltage in line with the IZH paper
	for PreNeurIdx in range(0, len(NeurArray)):
		if NeurArray[PreNeurIdx].jF:
			for PostNeurIdx in range(0, len(NeurArray)):
				NeurArray[PostNeurIdx].V += SynArray[PreNeurIdx][PostNeurIdx]*NeurArray[PreNeurIdx].exin

	return NeurArray

def zero_I(NeurArray):
	for NeurIdx in range(0, len(NeurArray)):
		NeurArray[NeurIdx].I = 0
	
	return NeurArray

def findOutput(NeurArray, SynArray):
	t = 0
	while t < sim_length:
		NeurArray = step_I(NeurArray, SynArray)
		NeurArray = step_v(NeurArray)

	
		for i in range(0, num_neurons):
			if NeurArray[i].jF:
				plt.plot(t, i, marker="o", markersize=1, markeredgecolor="green", markerfacecolor = "white")

		t += 1
	plt.show()
		


def testModel(ModBrain, NeurArray):
	brainScore = scoreNetwork(NeurArray, ModBrain)

	print(brainScore)


ArrayOfNeurs = []
brain = np.zeros((num_neurons, num_neurons))
for i in range(0, num_neurons):
	exin = 1 - 2*(np.random.rand()>0.8)
	current = (np.random.rand()>0.66)*curr_mult*np.random.rand()
	ArrayOfNeurs.append(myNeuron(-65, -13, 0.02, 0.2, -65, 4, current, False, exin))
	for j in range(0, num_neurons):
		if j == i:
			continue
		brain[i][j] = np.random.rand()*max_volt


findOutput(ArrayOfNeurs, brain)


