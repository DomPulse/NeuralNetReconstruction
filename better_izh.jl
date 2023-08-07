using PyPlot
using CSV
using DataFrames
using JLD

#i just realized this cant save or change the exin values
#i feel stupid
#:(
#make it do that and also be sparse


df = CSV.read("C:\\Users\\Dominic\\Documents\\julia projects\\NormalIRIS.csv", DataFrame)
curr_mult = 30
name = "BetterIZHexin"

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
num_models = 100
num_samps = 15
num_gens = 500
num_mutate = 3

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
	end

	return counts
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

function randomSynArray(num_neurons)
	S = zeros(num_neurons, num_neurons)
	S = (rand(Float64, num_neurons, num_neurons))*50
	for j in 1:num_neurons
		S[j, j] = 0
	end

	return S
end

function combineSynArray(Syn1, Syn2)

	return (Syn1+Syn2)/2
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
	if num_ones == 1
		if idxGuess == idxAns
			return 1
		else
			return 0
		end
	end
	return -1 #loses points for wrong score
end

function scoreNetwork(NeurArray, TrainList, SynArray)
	mmse = 0
	for idx in TrainList
		NeurArray = resetNet(NeurArray)
		ans = zeros(3)
		ans[df[idx, 5]] = 1
		guess = findOutput(NeurArray, idx, SynArray)
		
		if maximum(guess) != 0
			guess/=maximum(guess)
		end
		mmse -= rightGuess(guess, ans)/length(TrainList)
		mmse += (MSELoss(guess, ans))/length(TrainList)
	end
	return mmse
end

function rightGuessTest(ar1, ar2)
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
	if num_ones == 1 && idxAns == idxGuess
		return 1
	end
	return 0
end

function scoreNetworkTest(NeurArray, TrainList, SynArray)
	mmse = 0
	for idx in TrainList
		NeurArray = resetNet(NeurArray)
		ans = zeros(3)
		ans[df[idx, 5]] = 1
		guess = findOutput(NeurArray, idx, SynArray)
		if maximum(guess) != 0
			guess/=maximum(guess)
		end
		mmse -= rightGuessTest(guess, ans)/length(TrainList)

	end
	return mmse
end

function mutateSynArray(SynArray)
	#picks a random entry and changes it provided it is not a diagonal (which would self stimulate)
	#should make this a float but ehh
	thresh = 0.1
	i = rand(1:num_neurons)
	j = rand(1:num_neurons)
	if i == j
		SynArray = mutateSynArray(SynArray)
	else
		SynArray[i, j] = (rand(Float64))*50
		if SynArray[i, j] < thresh && SynArray[i, j] > -thresh
			SynArray[i, j] = 0
		end
	end

	#makes sure all the low values are zerod
	for i in 1:num_neurons
		for j in 1:num_neurons
			if SynArray[i, j] < thresh && SynArray[i, j] > -thresh
				SynArray[i, j] = 0
			end
		end
	end
	return SynArray
end

function newArrayOfModels(ModArray, winners)
	NewArrayOfModels = zeros(num_models, num_neurons, num_neurons)
	for brain in 1:num_models
		Sidx1 = rand(winners)
		Sidx2 = rand(winners)
		NewArrayOfModels[brain, :, :] = combineSynArray(ModArray[Sidx1, :, :], ModArray[Sidx2, :, :])
		for m in 1:num_mutate
			NewArrayOfModels[brain, :, :] = mutateSynArray(NewArrayOfModels[brain, :, :])
		end
	end
	return NewArrayOfModels
end

function testArrayOfModels(ModArray, NeurArray)
	lowScore = 1000
	idxKeep = 1
	for brain in 1:num_models
		brainScore = scoreNetworkTest(NeurArray, 1:150, ModArray[brain, :, :])
		if brainScore < lowScore
			idxKeep = brain
			lowScore = brainScore
		end
	end
	println(lowScore)
	return idxKeep
end

function saveNet(ModArray, NeurArray)
	idxKeep = testArrayOfModels(ModArray, NeurArray)

	println(ModArray[idxKeep, :, :])
	save("C:/Users/Dominic/Documents/julia projects/BetterIZHexin.jld", "winner", ModArray[idxKeep, :, :])

	#d = load("C:/Users/Dominic/Documents/julia projects/new_izh_long.jld")
	#println(d["winner"])
end

function Evo()
	#generates all the different arrays of synapses which determine behavior of the network
	ArrayOfModels = zeros(num_models, num_neurons, num_neurons)
	for s in 1:num_models
		ArrayOfModels[s, :, :] = randomSynArray(num_neurons)
		ArrayOfModels[s, :, :] = mutateSynArray(ArrayOfModels[s, :, :])
	end

	ArrayOfNeurs = []
	ArrayOfEXIN = []
	for i in 1:num_neurons
		exin = (rand() > 0.3)*2 - 1 #should give 1 or -1 for exitatory or inhibitory, maybe revert back to 0.2?
		if i <= 4 || i >= num_neurons - 3
			exin = 1
		end
		println(exin)
		append!(ArrayOfEXIN, exin)
		append!(ArrayOfNeurs, [Neur(-65, -13, 0.02, 0.2, -65, 4, 0, false, exin)])
	end
	save("C:/Users/Dominic/Documents/julia projects/BetterIZHexin.jld", "exin", ArrayOfEXIN)
	lowestOfAll = 1

	for gen in 1:num_gens
		
		tl = rand(1:150, num_samps)
		meanScore = 0
		lowScore = 0
		highScore = 0
		allScores = []
		for brain in 1:num_models
			brainScore = scoreNetwork(ArrayOfNeurs, tl, ArrayOfModels[brain, :, :])
			append!(allScores, brainScore)
		end
		meanScore = sum(allScores)/length(allScores)
		highScore = maximum(allScores)
		lowScore = minimum(allScores)
		winners = []
		offset = 0.1
		for idx in 1:num_models
			if allScores[idx] <= meanScore-offset || allScores[idx] == lowScore
				append!(winners, idx)
			end
		end

		if mod(gen, 25) == 0
			saveNet(ArrayOfModels, ArrayOfNeurs)
		end

		ArrayOfModels = newArrayOfModels(ArrayOfModels, winners)
		println(gen, " ", length(winners), " ", lowScore, " ", meanScore)
	end

	
end

Evo()

