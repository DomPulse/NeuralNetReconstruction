using CSV
using DataFrames
using JLD
using PyPlot

#inhibit very strongly each output neuron to each other
#regular STDP 
#raise voltage threshold to fire from whichever fires most
#https://www.youtube.com/watch?v=7TybETlCslM&ab_channel=EmbeddedSystemsWeek%28ESWEEK%29
#not a complete faithful implimentation of that, should try making everything fully connected, have multiple neurons per class, have a seperate inhibitory layer

#impliment 3 spike STDP
#have 3 broad layers, input layer, output layer of all exitatory, inhibitory layer, 4x9x9
#seems that only 2 layers are needed, input and ouput with the output acting to inhibit other outputs strongly? paper said 1 to 1 connection which seems pointless
#good weight normalization

df = CSV.read("D:\\Neuro Sci\\juliaIZH\\NormalIRIS.csv", DataFrame)
curr_mult = 30

mutable struct Neur
	V
	u
	a
	b
	c
	d
	I
	jF #just fired
	exin #exitatory or inhibitory
	fire_thresh #threshhold to fire, will only be modified on the output neurons
end

num_neurons = 4+15 #4 input neurons and 15 outputs, 5 for each class (ideally)
num_gens = 100
sim_length = 200
look_around = 20
max_volt = 20


function dif_v(v, u, I)

	return (0.04*v^2)+(5*v)+140-u+I
end

function dif_u(v, u, a, b)

	return a*(b*v-u)
end

function step_v(NeurArray)
	for NeurIdx in 1:length(NeurArray)
		NeurObj = NeurArray[NeurIdx]
		NeurObj.V += dif_v(NeurObj.V, NeurObj.u, NeurObj.I)
		NeurObj.u += dif_u(NeurObj.V, NeurObj.u, NeurObj.a, NeurObj.b)
		if NeurObj.jF == true
			NeurObj.jF = false
			NeurObj.V = NeurObj.c
			NeurObj.u = NeurObj.u + NeurObj.d
		end
		if NeurObj.V >= NeurObj.fire_thresh
			NeurObj.V = 30
			NeurObj.jF = true
		end

		if NeurObj.V <= -70
			NeurObj.V = -70
		end

		NeurArray[NeurIdx] = NeurObj
	end
	return NeurArray
end

function step_I(NeurArray, SynArray)
	#this formerly acted on the current input I 
	#has since been changed to act on voltage in line with the IZH paper
	for PreNeurIdx in 1:length(NeurArray)
		for PostNeurIdx in 1:length(NeurArray)
			bleh = SynArray[PreNeurIdx, PostNeurIdx]*NeurArray[PreNeurIdx].jF*NeurArray[PreNeurIdx].exin
			NeurArray[PostNeurIdx].V += bleh
		end
	end

	return NeurArray
end

function zero_I(NeurArray)
	for NeurIdx in 1:length(NeurArray)
		NeurArray[NeurIdx].I = 0
	end
	return NeurArray
end

function resetNet(NeurArray)
	resetSyns = zeros(num_neurons, num_neurons)
	NeurArray = zero_I(NeurArray)
	t = 0
	while t < 150
		NeurArray = step_v(NeurArray)
		NeurArray = step_I(NeurArray, resetSyns)

		t += 1
	end
	return NeurArray
end

function timeDifOfNear(PreSpikes, PostSpikes)
	sum_delay = 0
	n = 0
	for t in (look_around+1):(length(PreSpikes)-look_around) #searches presynaptic spikes, ignores early time before net could be expect to know
		PreTime = PreSpikes[t]*t
		if PreTime != 0
			triggered = false
			for tc in 1:look_around #searches within 5 time steps of the presynaptic spike
				PostTimeMinus = PostSpikes[t-tc]*(t-tc)
				PostTimePlus = PostSpikes[t+tc]*(t+tc) 
				if PostTimePlus != 0
					sum_delay += PreTime - PostTimePlus
					n += 1
					triggered = true
				end
				if PostTimeMinus != 0
					sum_delay += PreTime - PostTimeMinus
					n += 1
					triggered = true
				end
				if triggered
					break
					#this makes it so only the nearest
				end
			end
		end
	end
	if n == 0
		return 0
	end
	return (sum_delay/n)
end

function genSimOut(IrisIdx, NeurArray, SynArray, gen)
	Vs = zeros(num_neurons, sim_length)
	jFs = zeros(num_neurons, sim_length)

	t = 0
	NeurArray[1].I = df[IrisIdx, 1]*curr_mult
	NeurArray[2].I = df[IrisIdx, 2]*curr_mult
	NeurArray[3].I = df[IrisIdx, 3]*curr_mult
	NeurArray[4].I = df[IrisIdx, 4]*curr_mult

	while t < sim_length
		NeurArray = step_I(NeurArray, SynArray)
		NeurArray = step_v(NeurArray)		
		t += 1

		for i in 1:num_neurons
			Vs[i, t] = NeurArray[i].V
			jFs[i, t] = NeurArray[i].jF
		end
	end

	#=
	fig, axs = PyPlot.subplots(7)
	axs[1].plot(1:sim_length, Vs[1, :])
	axs[2].plot(1:sim_length, Vs[2, :])
	axs[3].plot(1:sim_length, Vs[3, :])
	axs[4].plot(1:sim_length, Vs[4, :])
	axs[5].plot(1:sim_length, Vs[num_neurons-2, :])
	axs[6].plot(1:sim_length, Vs[num_neurons-1, :])
	axs[7].plot(1:sim_length, Vs[num_neurons-0, :])
	PyPlot.show()
	=#

	return jFs
end

function normalizeSyn(SynArray, SparseArray)
	for i in 5:num_neurons
		num_connect = sum(SparseArray[:, i]) #number of connections going into this neuron
		sum_weights = sum(SynArray[:, i]) #sum of all weights going into this neuron
		
		for j in 1:num_neurons
			SynArray[j, i] *= num_connect*max_volt*0.5/sum_weights #slightly modified normalization equation from 
			#A biologically plausible supervised learning method for spiking neural networks using the symmetric STDP rule
		end
	end
	return SynArray
end

function randomSynArray(num_neurons)

	S = (rand(num_neurons, num_neurons))*max_volt
	return S
end

function genCustomSyns(num_neurons)
	SynArray = ones(num_neurons, num_neurons)
	for i in 1:num_neurons
		for j in 1:num_neurons
			if j <= 4
				SynArray[i, j] = 0 #no stimulation of input neurons
			end
			if i == j
				SynArray[i, j] = 0 #no self stimulation
			end
		end
	end
	return SynArray
end

function genAllowedSpace(ar1, dim)
	c = sum(ar1)
	c = trunc(Int, c)
	allowed = zeros(c, 2)
	idx = 0
	for i in 1:dim
		for j in 1:dim
			if ar1[i, j] > 0
				idx+=1
				allowed[idx, :] = [i, j]
			end
		end
	end
	return(allowed)
end

function STDP()
	SpArray = genCustomSyns(num_neurons)
	Brain = randomSynArray(num_neurons).*SpArray
	allowedSpaces = genAllowedSpace(SpArray, num_neurons)

	ArrayOfNeurs = []
	ArrayOfEXIN = []

	for i in 1:num_neurons
		if i <= 4
			exin = 1
		else
			exin = -1
		end
		append!(ArrayOfEXIN, exin)
		append!(ArrayOfNeurs, [Neur(-65, -13, 0.02, 0.2, -65, 4, 0, false, exin, -30)])
	end

	println(ArrayOfEXIN)
	save("D:/Neuro Sci/juliaIZH/STDPFullInhibitExin.jld", "exin", ArrayOfEXIN)

	l = length(allowedSpaces)/2
	learning_rate = 0.1

	learnDir = 1 #maybe just remove this?

	for gen in 1:num_gens
		delta = zeros(num_neurons, num_neurons)
		for pick in 1:150

			allSpikes = genSimOut(pick, ArrayOfNeurs, Brain, gen)

			counts = zeros(3)
			counts[3] = sum(allSpikes[num_neurons-2, :])
			counts[2] = sum(allSpikes[num_neurons-1, :])
			counts[1] = sum(allSpikes[num_neurons-0, :])

			ArrayOfNeurs = resetNet(ArrayOfNeurs)

			for idx in 1:l
				idx = trunc(Int64, idx)
				i, j = allowedSpaces[idx, :]
				i = trunc(Int64, i)
				j = trunc(Int64, j)

				t = timeDifOfNear(allSpikes[i, :], allSpikes[j, :])

				if t < 0
					delta[i, j] += learning_rate*exp(t/look_around)*ArrayOfNeurs[i].exin
				elseif t > 0
					delta[i, j] += learning_rate*exp(-t/look_around)*ArrayOfNeurs[i].exin
				end

			end
			ArrayOfNeurs = resetNet(ArrayOfNeurs)
		end

		Brain += delta
		Brain = normalizeSyn(Brain, SpArray)
		
		for i in 1:num_neurons
			for j in 1:num_neurons
				if Brain[i, j] > max_volt
					Brain[i, j] = max_volt
				elseif Brain[i, j] < 0
					Brain[i, j] = 0
				end
			end
		end

		println(gen)

	end
	println(Brain)
	save("D:/Neuro Sci/juliaIZH/STDPFullInhibitBrain.jld", "winner", Brain)
end

STDP()