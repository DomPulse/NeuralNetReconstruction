mutable struct Neur
	V
	u
	a
	b
	c
	d
	I
	jF
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
	for ResNeurIdx in 1:length(NeurArray)
		NeurArray[ResNeurIdx].I = rand(Float64)*10
	end
	for PreNeurIdx in 1:length(NeurArray)
		for PostNeurIdx in 1:length(NeurArray)
			NeurArray[PostNeurIdx].I += SynArray[PreNeurIdx, PostNeurIdx]*20*NeurArray[PreNeurIdx].jF
		end
	end

	return NeurArray
end

num_neurons = 50
ArrayOfNeurs = []
for i in 1:num_neurons
	append!(ArrayOfNeurs, [Neur(-65, -13, 0.02, 0.2, -65, 4, 0, false)])
end
ArrayOfNeurs[1].I = 10

#println(ArrayOfNeurs)
S = zeros(num_neurons, num_neurons)
S = rand(Float64, (num_neurons,num_neurons)) - 0.3*ones(num_neurons, num_neurons)
for j in 1:num_neurons
	S[j, j] = 0
end
#println(S)

t = 0
while t < 500
	#global ArrayOfNeurs[1].I += 10
	global ArrayOfNeurs = step_v(ArrayOfNeurs)
	global ArrayOfNeurs = step_I(ArrayOfNeurs, S)
	global t += 1

	for i in 1:num_neurons
		if ArrayOfNeurs[i].jF
			print("+")
		else
			print("-")
		end
	end
	println("")

end

