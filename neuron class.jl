using PyPlot
using CSV
using DataFrames
df = CSV.read("C:\\Users\\Dominic\\Documents\\julia projects\\NormalIRIS.csv", DataFrame)

mutable struct Neur
	V
	u
	a
	b
	c
	d
	I
	jF
	Vs
end

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
	#=
	#NeurObj = Neur(-65, -13, 0.02, 0.2, -65, 4, 10, false)
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

	return NeurObj
	=#
end

function step_I(NeurArray, SynArray)
	for PreNeurIdx in 1:length(NeurArray)
		for PostNeurIdx in 1:length(NeurArray)
			NeurArray[PostNeurIdx].I += SynArray[PreNeurIdx, PostNeurIdx]*1*NeurArray[PreNeurIdx].jF
		end
	end

	return NeurArray
end

function findOutput(NeurArray, IrisIdx, SynArray)
	counts = zeros(3)
	t = 0
	while t < 300

		NeurArray[1].I += df[IrisIdx, 1]
		NeurArray[2].I += df[IrisIdx, 2]
		NeurArray[3].I += df[IrisIdx, 3]
		NeurArray[4].I += df[IrisIdx, 4]

		
		NeurArray = step_v(NeurArray)
		NeurArray = step_I(NeurArray, S)
		for i in 1:num_neurons
			append!(NeurArray[i].Vs, NeurArray[i].V)
		end

		t += 1

		counts[1] += NeurArray[num_neurons-2].jF*1
		counts[2] += NeurArray[num_neurons-1].jF*1
		counts[3] += NeurArray[num_neurons-0].jF*1
	end
	if maximum(counts) != 0
		return counts/(maximum(counts))
	else
		return counts
	end
end

num_neurons = 20
ArrayOfNeurs = []
for i in 1:num_neurons
	append!(ArrayOfNeurs, [Neur(-65, -13, 0.02, 0.2, -65, 4, 0, false, [])])
end

#println(ArrayOfNeurs)
S = zeros(num_neurons, num_neurons)
S = rand(-5:5, (num_neurons,num_neurons))/3
for j in 1:num_neurons

	S[j, j] = 0
end
for j in 1:num_neurons
	for k in 1:num_neurons
		if S[j, k] < 1 && S[j, k] > -1
			S[j, k] = 0
		end
	end
end
#println(S)

t = 0
Ts = []


idx = 60
counts = zeros(3)

println(findOutput(ArrayOfNeurs, 60, S))

while t < 300

	global ArrayOfNeurs[1].I += df[idx, 1]
	global ArrayOfNeurs[2].I += df[idx, 2]
	global ArrayOfNeurs[3].I += df[idx, 3]
	global ArrayOfNeurs[4].I += df[idx, 4]

	
	global ArrayOfNeurs = step_v(ArrayOfNeurs)
	global ArrayOfNeurs = step_I(ArrayOfNeurs, S)
	for i in 1:num_neurons
		append!(ArrayOfNeurs[i].Vs, ArrayOfNeurs[i].V)
	end

	global t += 1
	append!(Ts, t)

	#=
	for i in num_neurons-2:num_neurons
		if ArrayOfNeurs[i].jF
			print("+")
		else
			print("_")
		end
	end
	println("")
	=#
	global counts[1] += ArrayOfNeurs[num_neurons-2].jF*1
	global counts[2] += ArrayOfNeurs[num_neurons-1].jF*1
	global counts[3] += ArrayOfNeurs[num_neurons-0].jF*1
end

println(counts)
println(df[idx, 1])

#=
fig, axs = PyPlot.subplots(num_neurons)
for i in 1:num_neurons
	axs[i].plot(Ts, ArrayOfNeurs[i].Vs)
end
PyPlot.show()
=#
