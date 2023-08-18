using PyPlot
using CSV
using DataFrames
using JLD

df = CSV.read("D:\\Neuro Sci\\juliaIZH\\NormalIRIS.csv", DataFrame)

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

curr_mult = 30
num_neurons = 20
num_out = 3
num_samps = 25
num_gens = 3000
sim_length = 300
max_volt = 50
T0 = 2.5

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
	#maybe initialize as all zeros rather than random values
	#maybe make these all integers?
	S = zeros(num_neurons, num_neurons)
	S = (rand(0:max_volt, num_neurons, num_neurons))
	return S
end

function genSparseArray(dim)
	SparseArray = ones(dim, dim)
	#println(SynArray)
	Sparse1 = rand([0,1], dim, dim)
	SparseArray .*= Sparse1
	for j in 1:num_neurons
		SparseArray[j, j] = 0 #ensures no self stimulation
		for i in 1:4
			SparseArray[j, i] = 0 #ensures nothing stimulates raw input, would be weird if i could think and increase volume or the like
		end
	end
	#println(SparseArray)
	return SparseArray
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

function mutateSynArray(SynArray)

	i = rand(1:num_neurons)
	j = rand(1:num_neurons)
	if i == j
		SynArray = mutateSynArray(SynArray)
	else
		SynArray[i, j] = (rand(Float64))*max_volt
	end

	return SynArray
end

function anneal()
	SpArray = genSparseArray(num_neurons)
	allowedSpaces = genAllowedSpace(SpArray, num_neurons)

	Brain = randomSynArray(num_neurons).*SpArray

	ArrayOfNeurs = []
	ArrayOfEXIN = []
	for i in 1:num_neurons
		exin = (rand() > 0.3)*2 - 1
		if i <= 4
			exin = 1
		elseif i >= num_neurons - 3
			exin = -1
		end
		append!(ArrayOfEXIN, exin)
		append!(ArrayOfNeurs, [Neur(-65, -13, 0.02, 0.2, -65, 4, 0, false, exin)])
	end

	println(ArrayOfEXIN)
	println(SpArray)
	save("D:/Neuro Sci/juliaIZH/AnnealExin.jld", "exin", ArrayOfEXIN)
	
	l = length(allowedSpaces)/2

	max_score = 0
	since_max = 0

	for gen in 1:num_gens
		T = T0 - gen*T0/num_gens
		num_right_mutt = 0
		num_right_virgin = 0
		vigrinBrain = Brain
		muttBrain = mutateSynArray(Brain).*SpArray
		for pick in 1:150

			allSpikesVirgin = genSimOut(pick, ArrayOfNeurs, vigrinBrain, gen)
			ArrayOfNeurs = resetNet(ArrayOfNeurs)
			allSpikesMutt = genSimOut(pick, ArrayOfNeurs, muttBrain, gen)
			ArrayOfNeurs = resetNet(ArrayOfNeurs)

			ans = zeros(3)
			ans[df[pick, 5]] = 1
			countsVirgin = zeros(3)
			countsMutt = zeros(3)
			#just figured out that this ran for 3000 gens and gave good results but allSpikesMutt was replaced with allSpikesVigin :/
			countsVirgin[1] = sum(allSpikesVirgin[num_neurons-2, :])
			countsVirgin[2] = sum(allSpikesVirgin[num_neurons-1, :])
			countsVirgin[3] = sum(allSpikesVirgin[num_neurons-0, :])
			countsMutt[1] = sum(allSpikesMutt[num_neurons-2, :])
			countsMutt[2] = sum(allSpikesMutt[num_neurons-1, :])
			countsMutt[3] = sum(allSpikesMutt[num_neurons-0, :])
			if maximum(countsVirgin) != 0
				countsVirgin/=(maximum(countsVirgin))
			end
			if maximum(countsMutt) != 0
				countsMutt/=(maximum(countsMutt))
			end
			num_right_virgin += rightGuess(countsVirgin, ans)
			num_right_mutt += rightGuess(countsMutt, ans)
			
		end

		if num_right_mutt > num_right_virgin
			Brain = muttBrain
		else
			if exp((num_right_mutt - num_right_virgin)/T) > rand()
				Brain = muttBrain
			end
		end

		since_max += 1
		if num_right_virgin > max_score
			max_score = num_right_virgin
			since_max = 0
			save("D:/Neuro Sci/juliaIZH/AnnealBrain.jld", "winner", vigrinBrain)
		end

		if since_max >= 60
			since_max = 0
			Brain = load("D:/Neuro Sci/juliaIZH/AnnealBrain.jld")["winner"]
		end
		
		println(gen, " ", num_right_virgin, " ", since_max, " ", max_score)
	end
	println(Brain)
	
	save("D:/Neuro Sci/juliaIZH/AnnealBrainFinal.jld", "winner", Brain)
end

anneal()