import numpy as np
import random
from sympy import *
import matplotlib.pyplot as plt
import copy



#this program will:
#generate a series of points near a specified line with some magnitude of random noise added
#generate a system of equations from these points
#generate a line of best fit from this system of equations using the gauss-newton algorithm

#generates all the points with some noise
num_of_points = 100
num_of_var = 3
m = 0.3
b = -0.1
l = 0.5
random_mag = 0.1
x_points = np.random.rand((num_of_points))
y_points = x_points*m + b + random_mag*(np.random.rand((num_of_points))-0.5) + l*x_points*x_points
approx_var = np.random.rand((num_of_var))

#generates all the functions needed for the system of equations
func_array = []
var_array = []
for var in range(0, num_of_var):
	var_array.append(Symbol("v"+str(var)))
for i in range(0, num_of_points):
	temp_func = -y_points[i] + x_points[i]*var_array[0] + var_array[1] + var_array[2]*x_points[i]*x_points[i]
	func_array.append(temp_func)
func_array = Matrix(func_array)
#print(func_array)

#the heavy lifting with the jacobians
#this step takes forever with trig functions present
#I should be able to make everything nice and linear by using the guessed values for wires to get numerical values for all nodes and propigate that way with no symbolic math
#derivatives and matrix inverses using arctan, forgetaboutit

Jacob = []
for i in range(0, num_of_points):
	to_append = []
	for j in range(0, num_of_var):
		deriv = diff(func_array[i], var_array[j])
		to_append.append(deriv)
	Jacob.append(to_append)
Jacob = Matrix(Jacob)
JacobT = Jacob.transpose()
JacJacT = JacobT*Jacob
InvJacJacT = JacJacT.inv() #idk this is weird for some reason
FinalMat = InvJacJacT*JacobT


#something isn't substiuting the variables correctly
#print(approx_var)
for k in range(0, 25):
	temp_func_array = copy.deepcopy(func_array)
	for f in range(0, len(temp_func_array)):
		subs_array = []
		for i in range(0, len(var_array)):
			subs_array.append((var_array[i], approx_var[i]))
		temp_func_array[f] = temp_func_array[f].subs(subs_array) #this doesn't quite work

	delta = FinalMat*temp_func_array
	del_array = []
	for d in delta:
		del_array.append(d)
	approx_var = approx_var - del_array
	print(del_array)

#print(approx_var)
#print(m, b, l)

recon_x = np.linspace(0, 1, 3)
recon_y = recon_x*approx_var[0]+approx_var[1]+approx_var[2]*recon_x*recon_x
plt.plot(recon_x, recon_y)
plt.scatter(x_points, y_points)
plt.show()

print(str(5)+"iljhnjjfghn")