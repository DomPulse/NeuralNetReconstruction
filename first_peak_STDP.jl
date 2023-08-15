using PyPlot
using CSV
using DataFrames
using JLD
using PyPlot

#just a reminder to try ant colony optimization whether or not this works
#hey there dumb dumb, if you think that theres something broken, you commented out the second sparse array so that's why there are so many entries

df = CSV.read("C:\\Users\\Dominic\\Documents\\julia projects\\NormalIRIS.csv", DataFrame)
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
end

num_neurons = 20
num_out = 3
num_samps = 25
num_gens = 100
sim_length = 300
look_around = 20
decision_time = look_around*2
max_volt = 30

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
	#hey there dumb dumb, if you think that theres something broken, you commented out the second sparse array so that's why there are so many entries
	SparseArray = ones(dim, dim)
	#println(SynArray)
	Sparse1 = rand(0:2, dim, dim)
	Sparse2 = rand(0:1, dim, dim)
	SparseArray .*= Sparse2#.*Sparse1
	#horrible way of tuning, should be able to pick % of desired 1's and just have it
	for j in 1:num_neurons
		SparseArray[j, j] = 0 #ensures no self stimulation
		for i in 1:4
			SparseArray[j, i] = 0 #ensures nothing stimulates raw input, would be weird if i could think and increase volume or the like
		end

		for i in num_neurons-2:num_neurons
			SparseArray[i, j] = 0 #ensures no output neurons are presynaptic
		end

		for i in 1:num_neurons
			if SparseArray[i, j] > 0
				SparseArray[i, j] = 1
			end
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

function genRefOut(IrisIdx)
	NeurArray = []
	for i in 1:num_out
		append!(NeurArray, [Neur(-65, -13, 0.02, 0.2, -65, 4, 0, false, 1)])
	end
	Vs = zeros(num_out, sim_length)
	jFs = zeros(num_out, sim_length)
	t = 0
	bleh = 0.5*ones(3)
	bleh[df[IrisIdx, 5]] = 1.5
	NeurArray[1].I = bleh[1]*curr_mult
	NeurArray[2].I = bleh[2]*curr_mult
	NeurArray[3].I = bleh[3]*curr_mult

	while t < sim_length
		NeurArray = step_v(NeurArray)

		t += 1

		for i in 1:num_out
			Vs[i, t] = NeurArray[i].V
			jFs[i, t] = NeurArray[i].jF
		end
	end

	#=
	fig, axs = PyPlot.subplots(3)
	axs[1].plot(1:sim_length, Vs[1, :])
	axs[2].plot(1:sim_length, Vs[2, :])
	axs[3].plot(1:sim_length, Vs[3, :])
	PyPlot.show()
	=#
	return jFs
end

function timeDifOfNear(PreSpikes, PostSpikes)
	sum_delay = 0
	n = 0
	for t in (look_around+decision_time+1):(length(PreSpikes)-look_around) #searches presynaptic spikes, ignores early time before net could be expect to know
		PreTime = PreSpikes[t]*t
		if PreTime != 0
			triggered = false
			for tc in 1:look_around #searches within 5 time steps of the presynaptic spike
				PostTimeMinus = PostSpikes[t-tc]*(t-tc)
				PostTimePlus = PostSpikes[t+tc]*(t+tc) 
				if PostTimePlus != 0
					sum_delay += PreTime - PostTimePlus
					n += 1
					#triggered = true
				end
				if PostTimeMinus != 0
					sum_delay += PreTime - PostTimeMinus
					n += 1
					#triggered = true
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

	#fpga paper kinda confirmed this as an idea to try at least, called a training signal
	#fpga paper is in computational intelligence and bioinspired systems
	genadj = (num_gens-gen)/num_gens
	#genadj = 1/(0.1*gen)

	bleh = -1.5*ones(3)
	bleh[df[IrisIdx, 5]] *= -1
	NeurArray[num_neurons-2].I = bleh[1]*curr_mult*genadj
	NeurArray[num_neurons-1].I = bleh[2]*curr_mult*genadj
	NeurArray[num_neurons-0].I = bleh[3]*curr_mult*genadj
		
	
	while t < sim_length
		sig = -1+(2/(1+exp(-0.05*(t-1*sim_length/3))))
		NeurArray[num_neurons-2].I = bleh[1]*curr_mult*genadj*sig
		NeurArray[num_neurons-1].I = bleh[2]*curr_mult*genadj*sig
		NeurArray[num_neurons-0].I = bleh[3]*curr_mult*genadj*sig
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

function STDP()
	SpArray = genSparseArray(num_neurons)
	allowedSpaces = genAllowedSpace(SpArray, num_neurons)

	Brain = randomSynArray(num_neurons).*SpArray
	OGSum = sum(Brain)

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
	save("C:/Users/Dominic/Documents/julia projects/STDPexinFirstDense.jld", "exin", ArrayOfEXIN)

	l = length(allowedSpaces)/2
	learning_rate = 1/150

	gen = 1
	stuckCount = 0
	bufferBrain = zeros(num_neurons, num_neurons)
	while gen < num_gens
		
		delta = zeros(num_neurons, num_neurons)
		num_right = 0
		for pick in 1:150
			allSpikes = genSimOut(pick, ArrayOfNeurs, Brain, gen)
			#refSpikes = genRefOut(pick)

			ans = zeros(3)
			ans[df[pick, 5]] = 1
			counts = zeros(3)
			counts[1] = sum(allSpikes[num_neurons-2, :])
			counts[2] = sum(allSpikes[num_neurons-1, :])
			counts[3] = sum(allSpikes[num_neurons-0, :])

			if maximum(counts) != 0
				counts/=maximum(counts)
			end
			num_right += rightGuess(counts, ans)

			

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
		if num_right < 135 #only moves on when accuracy still greater than 90%
			gen -= 1
			stuckCount += 1
		else
			stuckCount = 0
			bufferBrain = Brain
		end
		Brain += delta
		Brain += (rand(num_neurons, num_neurons) - 0.5*ones(num_neurons, num_neurons))*0.5.*SpArray
		#add random noise term to avoid being stuck in ruts like it is now

		#before the stuck counter and bufferBrain stuff this made it to gen 47 before effectively getting stuck in a loop
		if stuckCount > 15
			Brain = bufferBrain
			stuckCount = 0
		end
		
		for i in 1:num_neurons
			for j in 1:num_neurons
				if Brain[i, j] > max_volt
					Brain[i, j] = max_volt
				elseif Brain[i, j] < 0
					Brain[i, j] = 0
				end
			end
		end
		Brain *= OGSum/sum(Brain) #makes sure it doesn't blow up to all maxes and 0s
		println(gen, " ", stuckCount, " ", num_right)
		gen += 1
		
	end
	println(Brain)
	save("C:/Users/Dominic/Documents/julia projects/STDPbrainFirstDense.jld", "winner", Brain)
end

STDP()
