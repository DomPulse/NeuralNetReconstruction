import numpy as np
import matplotlib.pyplot as plt
import time


init_neuron = [-65.0, -13.0, 0.0, 0.0] #V, u, I, jF for the neuron at the start of sim
num_neurons = 1024
frac_connected = 1
sim_length = 1000
frac_inhib = 0.2
neurons = np.zeros((num_neurons, 4))
alphabet = np.zeros((num_neurons, 4))
glob_syn_array = np.zeros((num_neurons, num_neurons))
exin_array = np.zeros(num_neurons)

#creates the inital conditions and the fixed fit parameters for the neurons
for pre_syn_idx in range(0, num_neurons):
	if np.random.rand() < frac_inhib:
		exin_array[pre_syn_idx] = -1 
		ri = np.random.rand()
		alphabet[pre_syn_idx] = [0.02+0.08*ri, 0.25-0.05*ri, -65, 2]
		neurons[pre_syn_idx] = [-65.0, -65.0*alphabet[pre_syn_idx][1], 0.0, 0.0]
	else:
		exin_array[pre_syn_idx] = 1
		re = np.random.rand()
		alphabet[pre_syn_idx] = [0.02, 0.2, -65+15*re**2, 8-6*re**2]
		neurons[pre_syn_idx] = [-65.0, -65.0*alphabet[pre_syn_idx][1], 0.0, 0.0]

#creates the synaptic array connecting neurons
for post_syn_idx in range(0, num_neurons):
	for pre_syn_idx in range(0, num_neurons):
		
		glob_syn_array[post_syn_idx][pre_syn_idx] = np.random.rand()*exin_array[pre_syn_idx]*(1-0.5*(exin_array[pre_syn_idx]==1))*(np.random.rand() < frac_connected)
		if pre_syn_idx == post_syn_idx:
			glob_syn_array[post_syn_idx][pre_syn_idx] = 0

def update_neuron(neuron, fit_params):
	a = fit_params[0]
	b = fit_params[1]
	c = fit_params[2]
	d = fit_params[3]
	neuron[3] = 0
	neuron[2] = 0
	dv = 0.5*((0.04*(neuron[0]**2))+(5*neuron[0])+140-neuron[1]+neuron[2])
	neuron[0] += dv
	dv = 0.5*((0.04*(neuron[0]**2))+(5*neuron[0])+140-neuron[1]+neuron[2])
	neuron[0] += dv
	du = a*(b*neuron[0]-neuron[1])
	neuron[1] += du
	if neuron[0] >= 30:
		neuron[0] = c
		neuron[1] += d
		neuron[3] = 1
	return neuron

def noisy_current(exin_array):
	mask = (np.multiply(exin_array, np.ones(num_neurons))+1)/2
	currents_ex = np.multiply(5*np.random.normal(0, 1, num_neurons), mask)
	currents_in = np.multiply(2*np.random.normal(0, 1, num_neurons), np.ones(num_neurons)-mask)
	return currents_in + currents_ex

def update_current(loc_syn_array, fires):
	currents = np.matmul(loc_syn_array, fires)
	return currents

jFs = np.zeros((num_neurons, sim_length))
mask = (np.multiply(exin_array, np.ones(num_neurons))+1)/2
print("initialized")
start_time = time.time()
for t in range(0, sim_length):

	neurons[:] = update_neuron(neurons[:], alphabet[:])
	neurons[:, 2] = noisy_current(exin_array)
	neurons[:, 2] += update_current(glob_syn_array, neurons[:, 3])

	#neurons[:, 0] += 0.5*((0.04*(neurons[:, 0]**2))+(5*neurons[:, 0])+140-neurons[:, 1]+neurons[:, 2])
	#neurons[:, 0] += 0.5*((0.04*(neurons[:, 0]**2))+(5*neurons[:, 0])+140-neurons[:, 1]+neurons[:, 2])
	#neurons[:, 1] += alphabet[:, 0]*(alphabet[:, 1]*neurons[:, 0] - neurons[:, 1])
	#neurons[:, 3] = 0
	
	#try to improve this to see if it goes in "real time"
	#neurons_that_fired = np.where(neurons[:, 0] > 30)

	#for n in neurons_that_fired[0]:
	#	neurons[n][0] = alphabet[n][2]
	#	neurons[n][1] += alphabet[n][3]
	#	neurons[n][3] = 1 
	jFs[:, t] = neurons[:, 3]
	#currents_ex = np.multiply(5*np.random.normal(0, 1, num_neurons), mask)
	#currents_in = np.multiply(2*np.random.normal(0, 1, num_neurons), np.ones(num_neurons)-mask)
	#neurons[:, 2] = currents_in + currents_ex
	#neurons[:, 2] += np.matmul(glob_syn_array, neurons[:, 3])

print("--- %s seconds ---" % (time.time() - start_time))
print(neurons[0][0])
plt.imshow(jFs)
plt.show()

