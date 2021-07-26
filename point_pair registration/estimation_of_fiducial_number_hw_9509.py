import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


mean_landmark = 0.9722
mean_1 = 1.2
mean_2 = 1.5

stdev = 0.32


f_tre = [x * 0.01 for x in range(1,100,1)]
n_0 = [(mean_landmark**2 + stdev**2) / x for x in f_tre]
n_1 = [(mean_1**2 + stdev**2) / x for x in f_tre]
n_2 = [(mean_2**2 + stdev**2) / x for x in f_tre]
#n_1 = [1.0**2 / x for x in f_tre]
#n_2 = [1.2**2 / x for x in f_tre]
#n_3 = [1.5**2 / x for x in f_tre]


fiducial_array = [6, 10, 15]
tre_097 = [0.698, 0.547, 0.445]
tre_120 = [0.839, 0.671, 0.552]
tre_150 = [1.031, 0.827, 0.664]



fig = plt.figure()
plt.plot(f_tre, n_0, label='FLE = 0.97 mm (Fitz)', color = 'g')
plt.plot(f_tre, n_1, label='FLE = 1.20 mm (Fitz)', color = 'blue')
plt.plot(f_tre, n_2, label='FLE = 1.50 mm (Fitz)', color = 'r')

#plt.scatter(tre_097, fiducial_array,  label = 'FLE = 0.97 mm (experiment)', color = 'g')
#plt.scatter(tre_120, fiducial_array,  label = 'FLE = 1.20 mm (experiment)', color = 'blue')
#plt.scatter(tre_150, fiducial_array,  label = 'FLE = 1.50 mm (experiment)', color = 'r')


#plt.scatter(f_tre, n_3, label='FLE = 1.5 mm')
plt.ylim((4,28))
plt.xlim((0,0.6))
plt.xlabel('TRE (mm)')
plt.ylabel('Fiducial marker number')
plt.title('Number of the fiducial markers (Fitzpatrick estimation)')
plt.legend()
plt.show()