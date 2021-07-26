# functions for processing numpy array and matrix

import numpy as np


# Function Description
# Extract elements in an 1-D, 2-D or 3-D array based on conditions
# Conditions include <, > or ==
# Inputs:
#   1. array to be processed
#   2. threshold values for condition check (currently only support one single threshold)
#   3. background_value (elements doesn't satisfy the condition will set to this default value)
#   4. condition:
#           '0' for '=='
#           '-1' for '<"
#           '1' for '>'
#
# Returns:
#   1. idx which satisfy the condition
#   2. elements values (in 1-D array) which satisfy the condition
#   3. An array which has the same shape with input array and keeps all elements satisfying condition check
#      while sets other elements to background values.
#
# Example:
#   a = [1,2,3], background = 0, threhold = 2, condition = '-1'
#   The output will be
#       output[0] = [0]
#       output[1] = 1
#       output[2] = [1, 0, 0]

def extract_array_elements(array, threshold, background_value, condition = 0):
    tem = np.copy(array)
    tem_out = np.ones(tem.shape) * background_value
    tem_idx = []
    tem_extract = []
    if condition == 0:
        print('Finding element which is equal to the threshold')
        tem_idx = np.where(tem == threshold)
        tem_extract = tem[tem_idx]
    if condition == -1:
        print('Finding element which is less than the threshold')
        tem_idx = np.where(tem < threshold)
        tem_extract = tem[tem_idx]
    if condition == 1:
        print('Finding element which is larger than the threshold')
        tem_idx = np.where(tem > threshold)
        tem_extract = tem[tem_idx]
    tem_out[tem_idx] = tem[tem_idx]
    return tem_idx, tem_extract, tem_out


# Function used to merge all elements in a list
# The elements of a list are np matrices.
# Combine all matrices to make one matrix
def combine_elements_in_list(list):
    matrix = list[0]
    for i in range(1, len(list), 1):
        matrix = np.vstack((matrix, list[i]))
    return matrix


# Check which interval index a value is
# References: https://stackoverflow.com/questions/34798343/fastest-way-to-check-which-interval-index-a-value-is
# Only works for ascending order
def check_interval_idx_single_value(value, intervals):
    #print('interval is', intervals)
    if intervals[0] < intervals[-1]:
        idx = np.searchsorted(intervals, value, side="left")  # ascending order
    if intervals[0] > intervals[-1]:
        idx = intervals.size - np.searchsorted(intervals[::-1], value, side="right")   # descending order
    return idx


