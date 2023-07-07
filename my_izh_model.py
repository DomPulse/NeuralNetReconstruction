import numpy as np
import matplotlib.pyplot as plt

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

def dif_v(v, u, I):
	if I < 0:
		return 0
	r = (0.04*v**2)+(5*v)+140-u+I
	return(r)

def dif_u(v, u, a, b):
	r = a*(b*v-u)
	return(r)

def incr_volt(neuron):
	buffer_neuron = neuron
	v = buffer_neuron.v
	u = buffer_neuron.u
	a = buffer_neuron.a
	b = buffer_neuron.b
	c = buffer_neuron.c
	d = buffer_neuron.d
	I = buffer_neuron.I
	justFired = buffer_neuron.justFired
	dv = dif_v(v, u, I)
	du = dif_u(v, u, a, b)
	buffer_neuron.v += dv
	buffer_neuron.u += du
	buffer_neuron.tim += 1
		
	if justFired == True:
		buffer_neuron.v = c
		buffer_neuron.justFired = False

	if buffer_neuron.v >= 30:
		buffer_neuron.v = 30
		buffer_neuron.u += d
		
		buffer_neuron.justFired = True

	if buffer_neuron.v >= -50:
		buffer_neuron.tim = 0

	if buffer_neuron.v <= c-10:
		buffer_neuron.v = c-10

	buffer_neuron.I = 0
	return buffer_neuron

def Conducks(t, T):
	return (t/T)*(np.exp(1-t/T))

def incr_curr(pre_n, post_n, index):
	buffer_neuron = post_n
	if pre_n.exin == 1:
		buffer_neuron.I += 5*pre_n.gbases[index]*(-buffer_neuron.v)*Conducks(pre_n.tim, pre_n.taus[index])
	else:
		buffer_neuron.I += 5*pre_n.gbases[index]*(-buffer_neuron.v-65)*Conducks(pre_n.tim, pre_n.taus[index])
	return buffer_neuron

'''
Ts = [0]
n0 = myNeuron(taus = [5], gbases = [1], tims = [10000], I = 4)
n1 = myNeuron()
Vs = [n0.v]

for t in range(1, 1000):
	Ts.append(t)
	n0.I = 4
	n0 = incr_volt(n0)
	n1 = incr_volt(n1)
	n1 = incr_curr(n0, n1, 0)
	#n1.I = n0.conducks()[0]*(n0.v-n1.v)*(n0.v>n1.v)
	#print(n0.conducks())
	Vs.append(n1.v)

plt.plot(Ts, Vs)
plt.show()
'''

