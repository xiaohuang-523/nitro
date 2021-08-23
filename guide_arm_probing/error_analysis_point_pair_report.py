import numpy as np
import region_points
import plot as Yomiplot
import matplotlib.pyplot as plt
import Writers as Yomiwrite
import Readers as Yomiread
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


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

    # define figure for plotting
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')

    # define color map
    map = cm.get_cmap('gist_ncar', 512)
    print(map(range(12)))
    newcmp = ListedColormap(map(np.linspace(0.25, 0.75, 256)))

    # Set up the axes limits
    ax.axes.set_xlim3d(left=-40, right=40)
    ax.axes.set_ylim3d(bottom=-40, top=40)
    ax.axes.set_zlim3d(bottom=0, top=40)

    # Create axes labels
    ax.set_xlabel('X(mm)')
    ax.set_ylabel('Y(mm)')
    ax.set_zlabel('Z(mm)')

    # read tooth surface points and plot
    tooth_surface_file = "tooth_surface.txt"
    tooth_surface_points = Yomiread.read_csv(path + tooth_surface_file, 3, -1)
    Yomiplot.plot_3d_points(tooth_surface_points, ax, 'grey', alpha=0.3)

    # define targets and plot
    base = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\Edentulous registration error analysis\\"
    measurement_file = "Raw_data\\target_fiducial_measurements.txt"
    points = Yomiread.read_csv(base + measurement_file, 4, 10, flag=0)
    targets_original = points[0:4, 1:4]
    Yomiplot.plot_3d_points(targets_original, ax, 'purple')

    # Select method and plot
    N_REGION = 4
    tolerance = 0.5
    BIAS_MEAN = 1.0
    RAN_MEAN = 0
    GREEN = 0
    YELLOW = 0
    RED = 0

    # check each target for TRE color plot, target values 1, 2, 3, 4
    # set 0 to disable this function
    CHECK_TARGET = 0

    # set 1 to plot colormap based on highest TRE (95 value) among all four targets
    HIGH_OF_ALL = 1

    # set to 1 to normalize the color in colormap, minimum = 0.05, maximum = 1.75
    NORMALIZE_COLOR = 0
    MIN_COLOR = 0.05
    MAX_COLOR = 1.75

    # plot default fiducials
    if N_REGION == 5:
        print('checking for 5-region method')
        measurement_file = "Raw_data\\target_fiducial_measurements.txt"
        points = Yomiread.read_csv(base + measurement_file, 4, 10, flag=0)
        fiducials_zero = points[4:9, 1:4]
        fiducials_original = np.copy(fiducials_zero)
        fiducials_original[2, :] = fiducials_zero[3, :]
        fiducials_original[3, :] = fiducials_zero[2, :]
    elif N_REGION == 4:
        print('checking for 4-region method')
        measurement_file = "Raw_data\\target_fiducial_measurements_region_4.txt"
        points = Yomiread.read_csv(base + measurement_file, 4, -1)
        fiducials_original = points[:, 1:4]
    elif N_REGION == 3:
        print('checking for 3-region method')
        measurement_file = "Raw_data\\target_fiducial_measurements_region_3.txt"
        points = Yomiread.read_csv(base + measurement_file, 4, -1)
        fiducials_original = points[:, 1:4]
    Yomiplot.plot_3d_points(fiducials_original, ax, 'blue')

    # define separation planes and plot
    center_point = np.array([-0.096, 0, 0])
    x_min = -20
    x_max = 20
    y_min = 0
    y_max = 20
    z_min = 0
    z_max = 40
    if N_REGION == 5:
        plot_YZ_plane(x_min, x_max, y_min, y_max, z_min, z_max, 20 * np.pi / 180, ax)
        plot_YZ_plane(x_min, x_max, y_min, y_max, z_min, z_max, -20 * np.pi / 180, ax)
        plot_YZ_plane(x_min, x_max, y_min, y_max, z_min, z_max, 80 * np.pi / 180, ax)
        plot_YZ_plane(x_min, x_max, y_min, y_max, z_min, z_max, -80 * np.pi / 180, ax)

    elif N_REGION == 4:
        plot_YZ_plane(x_min, x_max, y_min, y_max, z_min, z_max, 0 * np.pi / 180, ax)
        plot_YZ_plane(x_min, x_max, y_min, y_max, z_min, z_max, 60 * np.pi / 180, ax)
        plot_YZ_plane(x_min, x_max, y_min, y_max, z_min, z_max, -60 * np.pi / 180, ax)

    elif N_REGION == 3:
        plot_YZ_plane(x_min, x_max, y_min, y_max, z_min, z_max, 45 * np.pi / 180, ax)
        plot_YZ_plane(x_min, x_max, y_min, y_max, z_min, z_max, -45 * np.pi / 180, ax)


    # read region points and plot
    #region2_file = "5_region_1.0_tolerance_points_region_2.txt"
    #region2 = Yomiread.read_csv(path + region2_file, 3, -1)
    #Yomiplot.plot_3d_points(region2, ax, 'yellow')

    # read color points and plot
    if GREEN == 1:
        green_file = np.str(N_REGION) + "_region_" + np.str(tolerance) + "_tolerance_green.txt"
        green_points = Yomiread.read_csv(path + green_file, 3, -1)
        Yomiplot.plot_3d_points(green_points, ax, 'green', alpha=0.8)
    if YELLOW == 1:
        yellow_file = np.str(N_REGION) + "_region_" + np.str(tolerance) + "_tolerance_yellow.txt"
        yellow_points = Yomiread.read_csv(path + yellow_file, 3, -1)
        Yomiplot.plot_3d_points(yellow_points, ax, 'yellow', alpha=0.8)
    if RED == 1:
        red_file = np.str(N_REGION) + "_region_" + np.str(tolerance) + "_tolerance_red.txt"
        red_points = Yomiread.read_csv(path + red_file, 3, -1)
        Yomiplot.plot_3d_points(red_points, ax, 'red', alpha=0.6)

    # plot with colormap (based on each target's 95% value)
    if CHECK_TARGET != 0:
        all_value95 = np.ones(4)
        all_points = np.ones(3)
        for region in range(N_REGION):
            #value_95_file = np.str(N_REGION) + "_region_" + np.str(tolerance) + "_tolerance_value_95_region_" + np.str(region + 1) + ".txt"
            points_file = np.str(N_REGION) + "_region_" + np.str(RAN_MEAN) + "_random_tolerance_and " + np.str(BIAS_MEAN) + "_bias_tolerance_points_region_" + np.str(region + 1) + ".txt"

            #points_file = np.str(N_REGION) + "_region_" + np.str(tolerance) + "_tolerance_points_region_" + np.str(region + 1) + ".txt"
            value_95_file = np.str(N_REGION) + "_region_" + np.str(RAN_MEAN) + "_random_tolerance_and " + np.str(BIAS_MEAN) + "_bias_tolerance_value_95_region_" + np.str(region + 1) + ".txt"

            value_95 = Yomiread.read_csv(path + value_95_file, 4, -1)
            points = Yomiread.read_csv(path + points_file, 3, -1)
            all_value95 = np.vstack((all_value95, value_95))
            all_points = np.vstack((all_points, points))

        all_value95 = all_value95[1:,:]
        all_points = all_points[1:,:]
        all_value95_check = all_value95[:, CHECK_TARGET - 1]
        if NORMALIZE_COLOR == 1:
            origin = np.zeros(3)
            all_points = np.vstack((all_points, origin))  # add artifical point at origin with color value 0 to set the minimum point in colormap
            all_value95_check = np.append(all_value95_check, MIN_COLOR) # set minimum color to 0.05
            all_points = np.vstack((all_points, origin))
            all_value95_check = np.append(all_value95_check, MAX_COLOR) # set maximum color to 1.75

        figure = Yomiplot.plot_3d_points_color_map(all_points, all_value95_check, ax)
        fig1.colorbar(figure)
        plt.title('Target ' + np.str(CHECK_TARGET) + ' error analysis with ' + np.str(N_REGION) + ' fiducials at ' + np.str(BIAS_MEAN) + " mm bias tolerance and " + np.str(RAN_MEAN) + " mm mean tolerance")

    # plot with colormap (based on the highest 95 value among all four targets)
    if HIGH_OF_ALL == 1:
        all_value95 = np.ones(4)
        all_points = np.ones(3)
        for region in range(N_REGION):
            #value_95_file = np.str(N_REGION) + "_region_" + np.str(tolerance) + "_tolerance_value_95_region_" + np.str(
            #    region + 1) + ".txt"
            points_file = np.str(N_REGION) + "_region_" + np.str(RAN_MEAN) + "_random_tolerance_and " + np.str(
                BIAS_MEAN) + "_bias_tolerance_points_region_" + np.str(region + 1) + ".txt"
            #points_file = np.str(N_REGION) + "_region_" + np.str(tolerance) + "_tolerance_points_region_" + np.str(
            #    region + 1) + ".txt"
            value_95_file = np.str(N_REGION) + "_region_" + np.str(RAN_MEAN) + "_random_tolerance_and " + np.str(
                BIAS_MEAN) + "_bias_tolerance_value_95_region_" + np.str(region + 1) + ".txt"

            print('file is', path + value_95_file)
            value_95 = Yomiread.read_csv(path + value_95_file, 4, -1)
            points = Yomiread.read_csv(path + points_file, 3, -1)
            all_value95 = np.vstack((all_value95, value_95))
            all_points = np.vstack((all_points, points))

        all_value95 = all_value95[1:, :]
        all_points = all_points[1:, :]
        all_value95_max = np.max(all_value95, axis=1)

        # the following code is used to normalize the color map. the color values always start from 0.05 to 1.05
        if NORMALIZE_COLOR == 1:
            origin = np.zeros(3)
            all_points = np.vstack((all_points, origin))  # add artifical point at origin with color value 0 to set the minimum point in colormap
            all_value95_max = np.append(all_value95_max, MIN_COLOR) # set minimum color to 0.05
            all_points = np.vstack((all_points, origin))
            all_value95_max = np.append(all_value95_max, MAX_COLOR) # set maximum color to 1.75

        figure = Yomiplot.plot_3d_points_color_map(all_points, all_value95_max, ax)
        fig1.colorbar(figure)
        plt.title(
            'Error analysis with ' + np.str(N_REGION) + ' fiducials at ' + np.str(
                tolerance) + " mm tolerance level (highest TRE among all 4 targets)")
        plt.title('Error analysis with ' + np.str(N_REGION) + ' fiducials at ' + np.str(
                BIAS_MEAN) + " mm bias tolerance and " + np.str(RAN_MEAN) + " mm mean tolerance (highest TRE among all 4 targets)")

    # check minimum and maximum 95% value
    # maximum_95_total = []
    # minimum_95_total = []
    # for i in range(N_REGION):
    #     value_95_file = np.str(N_REGION) + "_region_" + np.str(tolerance) + "_tolerance_value_95_region_" + np.str(i+1) + ".txt"
    #     value_95 = Yomiread.read_csv(path + value_95_file, 4, -1)
    #     maximum_95 = np.max(value_95,axis=0)
    #     maximum_95_total.append(maximum_95)
    #     minimum_95 = np.min(value_95, axis=0)
    #     minimum_95_total.append(minimum_95)
    #
    # max_95_output = path + np.str(N_REGION) + "_region_" + np.str(tolerance) + "_tolerance_value_max_95_total.txt"
    # min_95_output = path + np.str(N_REGION) + "_region_" + np.str(tolerance) + "_tolerance_value_min_95_total.txt"
    # Yomiwrite.write_csv_matrix(max_95_output, maximum_95_total)
    # Yomiwrite.write_csv_matrix(min_95_output, minimum_95_total)

    plt.show()