# Xiao Huang @ 07/14/2021
# Code used for box plotting in group
# Source can be found
# https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots
from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes
# hold,
import numpy as np
import Readers as Yomiread

# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    setp(bp['medians'][1], color='red')

    setp(bp['boxes'][2], color='green')
    setp(bp['caps'][4], color='green')
    setp(bp['caps'][5], color='green')
    setp(bp['whiskers'][4], color='green')
    setp(bp['whiskers'][5], color='green')
    setp(bp['medians'][2], color='green')

    setp(bp['boxes'][3], color='magenta')
    setp(bp['caps'][4], color='magenta')
    setp(bp['caps'][5], color='magenta')
    setp(bp['whiskers'][4], color='magenta')
    setp(bp['whiskers'][5], color='magenta')
    setp(bp['medians'][3], color='magenta')


def plot_boxplot_sample():
    # Some fake data to plot
    A = [[1, 2, 5,],  [7, 2], [2, 5, 8, 7]]
    B = [[5, 7, 2, 2, 5], [7, 2, 5], [2, 5, 8, 7]]
    C = [[3,2,5,7], [6, 7, 3], [2, 5, 8, 7]]

    fig = figure()
    ax = axes()

    # first boxplot pair
    bp = boxplot(A, positions = [1, 2, 3], widths = 0.6)
    setBoxColors(bp)

    # second boxplot pair
    bp = boxplot(B, positions = [5, 6, 7], widths = 0.6)
    setBoxColors(bp)

    # thrid boxplot pair
    bp = boxplot(C, positions = [9, 10, 11], widths = 0.6)
    setBoxColors(bp)

    # set axes limits and labels
    xlim(0,12)
    ylim(0,9)
    ax.set_xticklabels(['A', 'B', 'C'])
    ax.set_xticks([2, 6, 10])

    # draw temporary red and blue lines and use them to create a legend
    hB, = plot([1,1],'b-')
    hR, = plot([1,1],'r-')
    hG, = plot([1,1],'g-')
    legend((hB, hR, hG),('Apples', 'Oranges', 'New'))
    hB.set_visible(False)
    hR.set_visible(False)
    hG.set_visible(False)

    #savefig('boxcompare.png')
    show()


def plot_boxplot(data1, data2, data3, data4):
    # prepare the data for plotting.
    # For each group, the data is a list of several 1-D lists
    target = []
    group_labels = []
    for i in range(NUMBER_TARGETS):
        target.append([data1[:,i], data2[:,i], data3[:,i], data4[:,i]])
        group_labels.append('target ' + np.str(i+1))

    fig = figure()
    ax = axes()

    position_list = []
    ticks_list = []
    for j in range(NUMBER_TARGETS):
        position_list.append(j)
        ticks_list.append((NUMBER_ORIENTATIONS+1)/2 + NUMBER_ORIENTATIONS * j + 3*j)
        print('ticks_list is', ticks_list)

    for j in range(NUMBER_TARGETS):
    #for j in range(1):
        # add value to all elements in a list
        # https://www.geeksforgeeks.org/python-adding-k-to-each-element-in-a-list-of-integers/
        delta = NUMBER_ORIENTATIONS * j + 3*j+1
        print('delta is', delta)
        new_list = list(map(lambda x: x+delta, position_list))
        print('new_list is', new_list)
        bp = boxplot(target[j], positions = new_list, widths = 0.6)
        setBoxColors(bp)


    # set axes limits and labels
    xlim(-1, (NUMBER_ORIENTATIONS + 1) * NUMBER_TARGETS + 7)
    #xlim(0, 5)
    ylim(0.6,1.2)
    ax.set_xticklabels(group_labels)
    ax.set_xticks(ticks_list)

    #ax.set_xticklabels(['target 1'])
    #ax.set_xticks([2.5])

    ax.set_ylabel('TRE 95% values (mm)')

    # draw temporary red and blue lines and use them to create a legend
    hB, = plot([1,1],'b-')
    hR, = plot([1,1],'r-')
    hG, = plot([1,1],'g-')
    hP, = plot([1, 1], 'm-')
    #legend(group_labels)
    legend((hB, hR, hG, hP),('Isotropic Errors', 'Strategy 1', 'Strategy 2', 'Strategy 3'))
    hB.set_visible(False)
    hR.set_visible(False)
    hG.set_visible(False)
    hP.set_visible(False)

    ax.set_title('TRE with ' + np.str(BIAS_MEAN) + ' mm tolerance')
    #savefig('boxcompare.png')
    show()


# define global values
NUMBER_TARGETS = 4
NUMBER_ORIENTATIONS = 4

RAN_MEAN = 0
BIAS_MEAN = 0.75
N_REGION = 4

path = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\Edentulous registration error analysis\\Results\\"



# read the random noise result
all_value95 = np.ones(4)
for region in range(N_REGION):
    value_95_file = np.str(N_REGION) + "_region_" + np.str(BIAS_MEAN) + "_tolerance_value_95_region_" + np.str(region + 1) + ".txt"

    value_95 = Yomiread.read_csv(path + "Random noise\\" + value_95_file, 4, -1)
    all_value95 = np.vstack((all_value95, value_95))
all_value95_random_noise = all_value95[1:, :]
#all_value95_check.append(all_value95_random_noise)


all_value95_check = []
# read the biased errors
for j in range(3):
    all_value95 = np.ones(NUMBER_TARGETS)
    for region in range(N_REGION):
        value_95_file = np.str(N_REGION) + "_region_" + np.str(RAN_MEAN) + "_random_tolerance_and " + np.str(BIAS_MEAN) \
                        + "_bias_tolerance_value_95_region_" + np.str(region + 1) + ".txt"

        value_95 = Yomiread.read_csv(path + "Orientation" + np.str(j + 1) + "\\" + value_95_file, 4, -1)
        # remove outliers based on
        for row in value_95:
            # print('row', row)
            if (all(x < 3 for x in row)):
                all_value95 = np.vstack((all_value95, row))

    all_value95 = all_value95[1:, :]
    all_value95_check.append(all_value95)



plot_boxplot(all_value95_random_noise, all_value95_check[0], all_value95_check[1], all_value95_check[2])