import numpy as np 
import matplotlib.pyplot as plt

num_of_row = 3
num_of_col = 3

sensor_height = 0.01

SensorValues = np.zeros((num_of_row, num_of_col))

def bio_savt_step(dl, not_r): #all physical constants ignored like a chad
	r_hat = np.multiply(not_r, 1/np.sqrt(not_r[0]**2+not_r[1]**2+not_r[2]**2))
	#print(np.sqrt(r_hat[0]**2+r_hat[1]**2+r_hat[2]**2))
	B_vect = np.cross(dl, r_hat)
	r_mag = np.sqrt(not_r[0]**2+not_r[1]**2+not_r[2]**2)
	#print(np.sqrt(not_r[0]**2+not_r[1]**2+not_r[2]**2))
	B_vect = np.multiply(B_vect, 1/(r_mag*r_mag))
	return B_vect

def draw_sensor_array(array):
	for i in range(0, num_of_row):
		for j in range(0, num_of_col):
			x_pos = 9*i/(num_of_row-1)
			y_pos = 9*j/(num_of_col-1)
			mult = np.arctan(array[i][j]*1)/(np.pi/2)
			#print(i, j)
			if mult >= 0:
				Color = [1*mult, 0, 0]
			else:
				Color = [0, -1*mult, 0]
			plt.plot(x_pos, y_pos, marker="o", markersize=10, markeredgecolor="white", markerfacecolor=Color)

def get_sensor_val_contribute(wire_x0, wire_y0, wire_xf, wire_yf, wire_current, sens_x, sens_y):

	totalB = [0, 0, 0]
	#totalB = 0
	m = (wire_yf-wire_y0)/(wire_xf-wire_x0)
	x = wire_x0
	dx = 0.1 #arbitrary, sets resolution
	while x < wire_xf:

		y = (x-wire_x0)*m+wire_y0

		r = [sens_x-x, sens_y-y, sensor_height]
		dl = [dx, m*dx, 0]
		totalB = np.add(bio_savt_step(dl, r), totalB)
		x += dx
	return np.multiply(totalB, wire_current)

def eval_wire_contribute(wire_array):
	wire_x0 = wire_array[0][0]
	wire_y0 = wire_array[1][0]
	wire_xf = wire_array[0][1]
	wire_yf = wire_array[1][1]
	wire_current = wire_array[2]

	for i in range(0, num_of_row):
		for j in range(0, num_of_col):
			x_pos = 9*i/(num_of_row-1)
			y_pos = 9*j/(num_of_col-1)
			#print(i, j)

			SensorValues[i][j] += get_sensor_val_contribute(wire_x0, wire_y0, wire_xf, wire_yf, wire_current, x_pos, y_pos)[2] #gets z components of the 

def clear():
	for i in range(0, num_of_row):
		for j in range(0, num_of_col):
			SensorValues[i][j] = 0

def final_sens_val():

	return SensorValues.flatten()

