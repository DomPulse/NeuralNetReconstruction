using PyPlot
using CSV
using DataFrames
using JLD

df = CSV.read("C:\\Users\\Dominic\\Documents\\julia projects\\NormalIRIS.csv", DataFrame)
curr_mult = 30
exin_array = [1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1]

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
end

num_neurons = 20
num_models = 50
num_samps = 20
num_gens = 50
num_mutate = 10

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
		if NeurObj.V >= -30
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
			NeurArray[PostNeurIdx].V += SynArray[PreNeurIdx, PostNeurIdx]*NeurArray[PreNeurIdx].jF*NeurArray[PreNeurIdx].exin
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

function findOutput(NeurArray, IrisIdx, SynArray)
	counts = zeros(3)
	t = 0
	
	while t < 300
		NeurArray = zero_I(NeurArray)
		NeurArray[1].I += df[IrisIdx, 1]*curr_mult
		NeurArray[2].I += df[IrisIdx, 2]*curr_mult
		NeurArray[3].I += df[IrisIdx, 3]*curr_mult
		NeurArray[4].I += df[IrisIdx, 4]*curr_mult
		NeurArray = step_I(NeurArray, SynArray)
		NeurArray = step_v(NeurArray)
		
		t += 1

		counts[1] += NeurArray[num_neurons-2].jF
		counts[2] += NeurArray[num_neurons-1].jF
		counts[3] += NeurArray[num_neurons-0].jF

		#=
		for idx in 1:num_neurons
			if NeurArray[idx].jF
				print("+")
			else
				print("_")
			end
		end
		println("")
		=#

	end
	if maximum(counts) != 0
		println(counts)
		return counts/(maximum(counts))
	else
		return counts
	end
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


function MSELoss(ar1, ar2)
	mse = 0
	for idx in 1:length(ar1)
		mse += ((ar1[idx]-ar2[idx])^2)/length(ar1)
	end
	return mse
end

function rightGuess(ar1, ar2)
	idxAns = 4
	idxGuess = 0
	num_ones = 0
	for idx in 1:3
		if ar2[idx] == 1
			idxAns = idx
		end
		if ar1[idx] == 1
			idxGuess = idx
			num_ones += 1
		end
	end
	if idxGuess == idxAns && num_ones == 1
		return 1
	end
	return 0
end

function scoreNetwork(NeurArray, TrainList, SynArray)
	mmse = 0
	for idx in TrainList
		NeurArray = resetNet(NeurArray)
		ans = zeros(3)
		ans[df[idx, 5]] = 1
		guess = findOutput(NeurArray, idx, SynArray)
		score = rightGuess(guess, ans)
		print(guess)
		print(" ")
		print(ans)
		print(" ")
		println(score)
		mmse -= score/length(TrainList)
		#mmse += (MSELoss(guess, ans))/length(TrainList)
	end
	return mmse
end


function testModel(ModBrain, NeurArray)
	brainScore = scoreNetwork(NeurArray, 1:150, ModBrain[:, :])

	println(brainScore)
end

ArrayOfNeurs = []
for i in 1:num_neurons
	
	append!(ArrayOfNeurs, [Neur(-65, -13, 0.02, 0.2, -65, 4, 0, false, exin_array[i])])
	#append!(ArrayOfNeurs, [Neur(-65, -13, 0.02, 0.2, -65, 4, 0, false, 1)])
end

d = load("C:/Users/Dominic/Documents/julia projects/BetterIZHexin.jld")
#println(d["exin"])
ModBrain = d["winner"]

testModel(ModBrain, ArrayOfNeurs)