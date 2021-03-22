# This script is used to perform spline grid matching
# Divide the spline into pieces and apply only the translation to match the splines.
# by Xiao Huang, 03/22/2021

import numpy as np
import spinle_interpolation as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Readers as Yomiread



# Check which interval index a value is
# References: https://stackoverflow.com/questions/34798343/fastest-way-to-check-which-interval-index-a-value-is
# Only works for ascending order
def check_interval_idx(values, intervals):
    idx = []
    for value in values:
        #idx.append(bisect.bisect_left(intervals, value))
        #idx.append(np.searchsorted(intervals, value, side='left'))  # sort ascending order
        idx.append(intervals.size - np.searchsorted(intervals[::-1], value, side="right")) # sort descending order
    return idx


test_pc = Yomiread.read_csv("G:\\My Drive\\Project\\IntraOral Scanner Registration\\STL_pc - trial1\\spline_target_points.csv", 3, -1)
sample, fine = sp.spline_interpolation_3d(test_pc)

# find occlusal normal direction
delta = []
for i in range(3):
    delta.append(np.abs(np.max(fine[i]) - np.min(fine[i])))
normal_idx = np.where(np.asarray(delta) < 10)[0][0]

# Assume the full arch is placed in normal orientation
center_x = (fine[0][0] + fine[0][-1]) / 2
center_y = (fine[1][0] + fine[1][-1]) / 2

boundary = np.zeros(len(fine[0]))
for i in range(len(fine[0])):
    dy = fine[1][i] - center_y
    dx = fine[0][i] - center_x
    cos = dy / np.sqrt(dx**2 + dy**2)
    boundary[i] = cos

test = np.linspace(0,1.,8)

print('test is', test)
print('boundary is', boundary)
idx = check_interval_idx(test, boundary)
print('idx is', idx)



fig = plt.figure()
plt.scatter(range(len(boundary)), boundary, label = 'grid')

fig2 = plt.figure()
plt.scatter(fine[0], fine[1], label = 'fine')
plt.legend()
plt.show()
exit()

#yder = interpolate.splev(xnew, tck, der=1)

fig1 = plt.figure()
plt.scatter(sample[1,:], sample[2,:], color='g', label='sample grid')
plt.scatter(fine[1,:], fine[2,:], color='r', label='fine grid')
plt.legend()
plt.show()
