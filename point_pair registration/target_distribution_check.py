import numpy as np
import region_points
import plot as Yomiplot
import matplotlib.pyplot as plt
import Writers as Yomiwrite
import Readers as Yomiread
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import statistics_analysis as sa


# Plot 3D Boxes
# Only works for rectangular boxes
def plot_YZ_plane (xmin,xmax,ymin,ymax,zmin,zmax, Rz, fig_control):
    # Plot surface in 3d
    # Source code can be checked from https://matplotlib.org/3.1.0/gallery/mplot3d/surface3d.html
    # Details about the function from https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
    X = np.arange(xmin, xmax, (xmax-xmin)/15)
    Y = np.arange(ymin, ymax, (ymax-ymin)/15)
    Z = np.arange(zmin, zmax, (zmax-zmin)/15)
    # plot Y-Z plane
    y, z = np.meshgrid(Y,Z)
    x = y * np.tan(Rz)
    # top surface
    fig_control.plot_surface(x, y, z, alpha = 0.1, color = 'grey')


if __name__ == "__main__":
    path = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\Edentulous registration error analysis\\Results\\"

    # Select method and plot
    N_REGION = 4
    tolerance = 1.0
    BIAS_MEAN = 0.75
    RAN_MEAN = 0
    ORIENTATION = 3

    # check each target for TRE color plot, target values 1, 2, 3, 4
    # set 0 to disable this function
    CHECK_TARGET = 4

    # define a figure obj for plotting
    figure1 = plt.figure()


    # plot MAXIMUM 95% TRE distribution
    if CHECK_TARGET != 0:
        for ORIENTATION in range(3):
            all_value95 = np.ones(4)
            all_points = np.ones(3)
            for region in range(N_REGION):
                value_95_file = np.str(N_REGION) + "_region_" + np.str(RAN_MEAN) + "_random_tolerance_and " + np.str(BIAS_MEAN) + "_bias_tolerance_value_95_region_" + np.str(region + 1) + ".txt"

                value_95 = Yomiread.read_csv(path + "Orientation"+np.str(ORIENTATION+1)+"\\" + value_95_file, 4, -1)

                # remove outliers based on
                for row in value_95:
                    #print('row', row)
                    if (all(x < 3 for x in row)):
                        all_value95 = np.vstack((all_value95, row))

            all_value95 = all_value95[1:,:]
            all_value95_check = all_value95[:, CHECK_TARGET - 1]

            y, bins, p = plt.hist(all_value95_check, bins=100, density=True, alpha=0.6, label='orientation ' + np.str(ORIENTATION+1))
            #plt.show()

    # read the random noise result
    all_value95 = np.ones(4)
    for region in range(N_REGION):
        value_95_file = np.str(N_REGION) + "_region_" + np.str(BIAS_MEAN) + "_tolerance_value_95_region_" + np.str(region + 1) + ".txt"
        value_95 = Yomiread.read_csv(path + "Random noise\\" + value_95_file, 4, -1)
        all_value95 = np.vstack((all_value95, value_95))
    all_value95 = all_value95[1:, :]
    all_value95_check = all_value95[:, CHECK_TARGET - 1]
    y, bins, p = plt.hist(all_value95_check, bins=100, density=True, alpha=0.6, label='random noise')


        #sa.fit_models(all_value95_check)
    #plt.xlim([0, 1])
    plt.xlabel('TRE 95 percent value (mm)')
    plt.title('95% TRE distribution on target' + np.str(CHECK_TARGET) + ' with ' + np.str(N_REGION) + 'fiducials at tolerance of ' + np.str(BIAS_MEAN) + ' mm')
    plt.legend()
    plt.show()