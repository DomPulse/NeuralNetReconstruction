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

function countsMax(allSpikes)
	count = zeros(15)
	for i in 1:15
		count[i] = sum(allSpikes[num_neurons-(i-1), :])
	end
	max = zeros(15)
	for i in 1:15
		if count[i] == maximum(count)
			max[i] = 1
		end
	end
	avg = sum(count)/15
	if avg <= 7.5
		println(1)
	elseif avg >= 10
		println(3)
	else
		println(2)
	end
end
			

function testModel(ModBrain, NeurArray)
	for pick in 1:150
		allSpikes = genSimOut(pick, ArrayOfNeurs, ModBrain, 1)
		print(pick, " ")
		countsMax(allSpikes)
	end
end

ArrayOfNeurs = []
k = load("D:\\Neuro Sci\\juliaIZH\\STDPFullInhibitExin.jld")
exin_array = k["exin"]

for i in 1:num_neurons
	
	append!(ArrayOfNeurs, [Neur(-65, -13, 0.02, 0.2, -65, 4, 0, false, exin_array[i], -30)])
	#append!(ArrayOfNeurs, [Neur(-65, -13, 0.02, 0.2, -65, 4, 0, false, 1)])
end

d = load("D:\\Neuro Sci\\juliaIZH\\STDPFullInhibitBrain.jld")
#println(d["exin"])
ModBrain = d["winner"]
println(ModBrain)

testModel(ModBrain, ArrayOfNeurs)