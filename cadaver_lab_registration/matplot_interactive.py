import numpy as np
import matplotlib.pyplot as plt
import DICOM_reader as DR
from bitarray import bitarray

# import slider bar in matplot
from matplotlib.widgets import Slider, Button

import point_cloud_manipulation as pcm
import plot as Yomiplot

import tkinter as tk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import gui

# draw a circle in figure.
def draw_circle(point1, point2):
    center = point1
    radius = np.linalg.norm(np.asarray(point1) - np.asarray(point2))
    tolerance = 0.9
    xmin = np.int(center[0] - radius)
    xmax = np.int(center[0] + radius)
    ymin = np.int(center[1] - radius)
    ymax = np.int(center[1] + radius)
    circle = []
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            dis_tem = np.linalg.norm(np.asarray(point1) - np.asarray((x,y)))
            if np.abs(dis_tem - radius) < tolerance:
                circle.append((x,y))
    return circle, radius


def draw_line(point1, point2):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    xmin = np.min([x1, x2])
    xmax = np.max([x1, x2])
    line = []
    for x in range(xmin, xmax):
        y = (y2 - y1)/(x2 - x1) * (x - x1) + y1
        line.append((int(x), int(y)))
    return line


# ct img class
class CTImg():
    def __init__(self, ct_file_path):
        self.file_path = ct_file_path
        self.scan, self.slice_properties = DR.load_scan_human(self.file_path)
        self.fig = None
        self.ax = None
        self.fig2 = None

        self.pixel_array = None
        self.pixel_array_tem = None
        self.click_points = []
        self.line_tem = []
        self.input_flag = True
        self.ix = 0
        self.iy = 0
        self.z_min = None
        self.z_max = None
        self.tooth_number = None
        self.tooth_surface = None

        # SRG method
        self.seed = []
        self.search_radius = []
        self.seed_selection_figure = None

        # Box method
        self.corner_list = []
        for i in range(32):
            self.corner_list.append([])
        self.corner_points = []
        self.box_selection_figure = None
        self.corner_list_o = []
        self.corner_list_l = []
        self.corner_list_b = []
        self.corner_list_f = []
        self.corner_list_bk = []
        for i in range(32):
            self.corner_list_o.append([])
            self.corner_list_l.append([])
            self.corner_list_b.append([])
            self.corner_list_f.append([])
            self.corner_list_bk.append([])
        self.corner_list_o_label = np.zeros(32)
        self.corner_list_l_label = np.zeros(32)
        self.corner_list_b_label = np.zeros(32)
        self.corner_list_f_label = np.zeros(32)
        self.corner_list_bk_label = np.zeros(32)
        self.total_corner_list = [self.corner_list_o, self.corner_list_l, self.corner_list_b, self.corner_list_f,
                                  self.corner_list_bk]
        self.total_corner_list_label = [self.corner_list_o_label, self.corner_list_l_label, self.corner_list_b_label,
                                        self.corner_list_f_label, self.corner_list_bk_label]

        # GUI method
        self.check_list_entry_o = []
        self.check_list_entry_l = []
        self.check_list_entry_b = []
        self.check_list_entry_f = []
        self.check_list_entry_bk = []
        self.total_check_list_entry = [self.check_list_entry_o, self.check_list_entry_l, self.check_list_entry_b, self.check_list_entry_f, self.check_list_entry_bk]
        self.total_check_list_entry_name = ['occlusal', 'lingual', 'buccal', 'front', 'back']
        self.total_check_list_entry_short_name = ['o', 'l', 'b', 'f', 'bk']
        self.tk_master = None
        self.e1 = None
        self.e2 = None
        self.e3 = None
        self.e4 = None
        self.step2_info = None
        self.step3_info = None

        self.generate_gui()

    def generate_gui(self):
        self.tk_master = tk.Tk()

        # set 4 rows and 2 columns
        for i in range(21):
            self.tk_master.rowconfigure(i, minsize=50, weight=1)
            for j in range(3):
                self.tk_master.columnconfigure(j, minsize=1, weight=4)
            for j in range(3,19,1):
                self.tk_master.columnconfigure(j, minsize=1, weight=1)

        # define entry and label for reading
        lbl1 = tk.Label(master = self.tk_master, text = "tooth number")
        lbl1.grid(row=3, column=1, sticky="nsew", padx=5, pady=5)
        self.e1 = tk.Entry(self.tk_master)
        self.e1.grid(row=3, column=2, sticky="nsew", padx=5, pady=5)
        self.e1.insert(0, 'Enter Value')
        lbl1_info = tk.Label(master=self.tk_master, text="Upper right-left is 1-16\n Lower right-left is 32-17")
        lbl1_info.grid(row=3, column=3, columnspan=5, sticky="nsew", padx=5, pady=5)

        lbl2 = tk.Label(master = self.tk_master, text = "tooth surface")
        lbl2.grid(row=4, column=1, sticky="nsew", padx=5, pady=5)
        self.e2 = tk.Entry(self.tk_master)
        self.e2.grid(row=4, column=2, sticky="nsew", padx=5, pady=5)
        self.e2.insert(0, 'Enter Value')
        lbl2_info = tk.Label(master=self.tk_master, text="Occlusal 'o', Lingual 'l', Buccal 'b'\n 'Front 'f', Back 'bk'")
        lbl2_info.grid(row=4, column=3, columnspan=5, sticky="nsew", padx=5, pady=5)

        lbl3 = tk.Label(master = self.tk_master, text = "Minimum Z value")
        lbl3.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self.e3 = tk.Entry(self.tk_master)
        self.e3.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)
        self.e3.insert(0, 'Enter Value')
        lbl4 = tk.Label(master = self.tk_master, text = "Maximum Z value")
        lbl4.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)
        self.e4 = tk.Entry(self.tk_master)
        self.e4.grid(row=2, column=2, sticky="nsew", padx=5, pady=5)
        self.e4.insert(0, 'Enter Value')
        lbl3_info = tk.Label(master=self.tk_master, text="Define the height range to seach for tooth surface")
        lbl3_info.grid(row=1, rowspan=2, column=3, columnspan=5, sticky="nsew", padx=5, pady=5)

        self.step2_info = tk.Label(master=self.tk_master, text="")
        self.step2_info.grid(row=5, column=2, columnspan = 5, sticky="nsew", padx=5, pady=5)

        self.step3_info = tk.Label(master=self.tk_master, text="")
        self.step3_info.grid(row=6, rowspan=2, column=2, columnspan = 5, sticky="nsew", padx=5, pady=5)

        # define plot function
        button = tk.Button(master=self.tk_master, command=self.plot_ct_slices, text="Step 1 \n plot CT slice overview")
        button.grid(row=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        button_2 = tk.Button(master=self.tk_master, command=self.update_z_values, text="Step 2 \n update Z values")
        button_2.grid(row=5, columnspan=2, sticky="nsew", padx=5, pady=5)

        button_3 = tk.Button(master=self.tk_master, command=self.update_box_values, text="Step 3 \n select box for segmentation")
        button_3.grid(row=6, columnspan=2, sticky="nsew", padx=5, pady=5)

        #button_4 = tk.Button(master=self.tk_master, command=self.update_box_list(), text="Step 4 \n update box points")
        #button_4.grid(row=7, columnspan=2, sticky="nsew", padx=5, pady=5)

        button_5 = tk.Button(master=self.tk_master, command=self.check_segment, text="Step 5 \n check segmented teeth")
        button_5.grid(row=10, columnspan=2, sticky="nsew", padx=5, pady=5)

        button_6 = tk.Button(master=self.tk_master, command=self.update_z_values, text="Step 6 \n perform tooth segmentation")
        button_6.grid(row=20, columnspan=2, sticky="nsew", padx=5, pady=5)

        for j in range(5):
            tk.Label(master=self.tk_master, text= self.total_check_list_entry_name[j]).grid(row=9+j, column=2, sticky="nsew", padx=5, pady=5)
            tk.Label(master=self.tk_master, text=self.total_check_list_entry_name[j]).grid(row=15 + j, column=2,
                                                                                           sticky="nsew", padx=5,
                                                                                           pady=5)
            for i in range(16):
                entry_upper = tk.Entry(self.tk_master, width=5)
                self.total_check_list_entry[j].append(entry_upper)
                self.total_check_list_entry[j][i].grid(row=9+j, column = i + 3, sticky="nsew", padx=5, pady=5)
            for i in range(16, 32, 1):
                entry_lower = tk.Entry(self.tk_master, width=5)
                self.total_check_list_entry[j].append(entry_lower)
                self.total_check_list_entry[j][i].grid(row=15+j, column= 32 - i + 2, sticky="nsew", padx=5, pady=5)

        for i in range(16):
            tk.Label(master=self.tk_master, text=np.str(i + 1), width=5).grid(row=8, column=i + 3, sticky="nsew",
                                                                              padx=5, pady=5)
        for i in range(16,32,1):
            tk.Label(master=self.tk_master, text=np.str(i + 1), width=5).grid(row=14, column=32 - i + 2, sticky="nsew",
                                                                              padx=5, pady=5)

        self.tk_master.mainloop()

    # check how many teeth are segmented
    def check_segment(self):
        for j in range(5):
            for i in range(32):
                if self.total_corner_list_label[j][i] == 1:
                    #self.total_check_list_entry[j][i].insert(0, 1)
                    self.total_check_list_entry[j][i].config({"background": "Green"})
                else:
                    self.total_check_list_entry[j][i].config({"background": "Red"})

    # update z values
    def update_z_values(self):
        self.z_min = int(self.e3.get())
        self.z_max = int(self.e4.get())
        self.step2_info.config(text='Tooth is segmented between ' + np.str(self.z_min) + ' and ' + np.str(self.z_max))
        if plt.fignum_exists('tooth segmentation figure'):
            self.fig2.clear()
            plt.close(self.fig2)
        self.fig2 = plt.figure('tooth segmentation figure')
        self.read_ct(self.z_min)
        self.select_box(self.fig2)


    # update box points
    def update_box_values(self):
        self.tooth_number = int(self.e1.get()) - 1
        self.tooth_surface = self.e2.get()
        if self.tooth_surface in self.total_check_list_entry_short_name:
            surface_idx = self.total_check_list_entry_short_name.index(self.tooth_surface)
        self.total_corner_list[surface_idx][self.tooth_number] = self.corner_points.copy()
        self.total_corner_list_label[surface_idx][self.tooth_number] = 1
        self.step3_info.config(text = 'Tooth '+np.str(self.tooth_number + 1) + ' '
                                      + self.total_check_list_entry_name[surface_idx]
                                      + ' surface is segmented in \n box '
                                      + np.str(self.total_corner_list[surface_idx][self.tooth_number])
                                      + ' \nbetween slice '+ np.str(self.z_min) + ' and slice ' + np.str(self.z_max)
                               )

    def update(self, val):
        self.ax.clear()
        self.ax.imshow(self.scan[np.int(val)].pixel_array)
        # fig.canvas.draw()
        #self.fig.canvas.draw_idle()

    def plot_ct_slices(self):
        if plt.fignum_exists('slice overview figure'):
            plt.close(self.fig)
        self.fig, self.ax = plt.subplots(num = 'slice overview figure')
        init_slice = 35
        img = self.scan[init_slice].pixel_array
        HU = img * self.slice_properties[-2] + self.slice_properties[-1]
        plt.imshow(HU)

        self.ax.set_xlabel('Slice')
        axcolor = 'lightgoldenrodyellow'
        self.ax.margins(x=0)
        plt.subplots_adjust(left=0.25, bottom=0.25)

        # Make a horizontal oriented slider to control the slice
        axslice = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        slice_slider = Slider(
            ax=axslice,
            label="Slice",
            valmin=0,
            valmax=self.slice_properties[2],
            valinit=init_slice
        )
        slice_slider.on_changed(self.update)
        plt.show()

    # SRG method select seed
    def select_seed(self, slice_number, figure_control):
        img = self.scan[slice_number].pixel_array
        HU = img * self.slice_properties[-2] + self.slice_properties[-1]
        self.pixel_array = HU
        self.pixel_array_tem = np.copy(self.pixel_array)

        self.seed.clear()
        self.search_radius.clear()
        self.click_points.clear()
        self.seed_selection_figure = figure_control
        plt.imshow(self.pixel_array)
        print('Select seed points and radius in figure')
        self.seed_selection_figure.canvas.mpl_connect('button_press_event', self.onclick_srg)
        plt.show()

    # SRG method onclick event
    def onclick_srg(self, event):
        print('event is', event)
        self.ix, self.iy = event.xdata, event.ydata
        print('x = %d, y = %d' % (self.ix, self.iy))
        self.click_points.append((np.int(self.ix), np.int(self.iy)))
        if len(self.click_points) % 2 == 0:
            self.seed.clear()
            self.search_radius.clear()
            for i in range(len(self.click_points) // 2):
                circle_points, radius = draw_circle(self.click_points[i * 2], self.click_points[i * 2 + 1])
                self.seed.append(self.click_points[i * 2])
                self.search_radius.append(radius)
                for point in circle_points:
                    self.pixel_array_tem[point[1], point[0]] = 3000
        plt.imshow(self.pixel_array_tem)
        self.seed_selection_figure.canvas.draw()

    def clear_figure(self):
        self.fig.clear()
        plt.close(self.fig)

    def read_ct(self, slice_number):
        img = self.scan[slice_number].pixel_array
        HU = img * self.slice_properties[-2] + self.slice_properties[-1]
        self.pixel_array = HU
        self.pixel_array_tem = np.copy(self.pixel_array)

    # box method select corners
    def select_box(self, figure_control):
        self.click_points.clear()
        self.line_tem.clear()
        self.box_selection_figure = figure_control
        plt.imshow(self.pixel_array_tem)
        print('Select box corners in the figure')
        self.box_selection_figure.canvas.mpl_connect('button_press_event', self.onclick_box)
        plt.show()

    # box method onclick event
    def onclick_box(self, event):
        print('event is', event)
        self.ix, self.iy = event.xdata, event.ydata
        print('x = %d, y = %d' % (self.ix, self.iy))
        point_tem = (np.int(self.ix), np.int(self.iy))
        self.click_points.append(point_tem)
        if len(self.click_points) == 4:
            self.corner_points.clear()
            for i in range(4):
                if i < 3:
                    line = draw_line(self.click_points[i], self.click_points[i+1])
                    self.line_tem.append(line)
                    for point in line:
                        self.pixel_array_tem[point[1], point[0]] = 3000
                else:
                    line = draw_line(self.click_points[3], self.click_points[0])
                    self.line_tem.append(line)
                    for point in line:
                        self.pixel_array_tem[point[1], point[0]] = 3000
            self.corner_points = self.click_points.copy()
        elif len(self.click_points) > 4:
            # update the point in box
            point_idx = self.update_closed_point(point_tem)
            # find the old line idx
            line_idx_1 = point_idx - 1
            if line_idx_1 < 0:
                line_idx_1 = 3
            line_idx_2 = point_idx
            line_idx_3 = point_idx + 1
            if line_idx_3 > 3:
                line_idx_3 = 0

            # reset the old line color
            for point in self.line_tem[line_idx_1]:
                self.pixel_array_tem[point[1], point[0]] = self.pixel_array[point[1], point[0]]
            for point in self.line_tem[line_idx_2]:
                self.pixel_array_tem[point[1], point[0]] = self.pixel_array[point[1], point[0]]
            # update the new line list and draw color
            line_1 = draw_line(self.corner_points[line_idx_1], self.corner_points[line_idx_2])
            line_2 = draw_line(self.corner_points[line_idx_2], self.corner_points[line_idx_3])
            self.line_tem[line_idx_1] = line_1
            self.line_tem[line_idx_2] = line_2
            for point in self.line_tem[line_idx_1]:
                self.pixel_array_tem[point[1], point[0]] = 3000
            for point in self.line_tem[line_idx_2]:
                self.pixel_array_tem[point[1], point[0]] = 3000

        print('self.corner points are', self.corner_points)
        plt.imshow(self.pixel_array_tem)
        self.box_selection_figure.canvas.draw()

    # Box method update closed point
    def update_closed_point(self, point_tem):
        if self.corner_points == []:
            print('nothing to update')
            min_dis_idx = 0
        else:
            dis = []
            for point in self.corner_points:
                dis.append(np.linalg.norm(np.asarray(point_tem) - np.asarray(point)))
            min_dis = min(dis)
            min_dis_idx = dis.index(min_dis)
            self.corner_points[min_dis_idx] = point_tem
        return min_dis_idx

    # Box method update selected box points list
    def update_box_list(self):
        self.corner_list.append(self.corner_points)


# function to get intensity
def get_intensity(scan, coordinate): # coordinate is real [x, y, z]
    z_tem = coordinate[2]
    y_tem = coordinate[1]
    x_tem = coordinate[0]
    intensity = scan[z_tem].pixel_array[y_tem, x_tem]
    return intensity


# function to get intensity values from CT scan
def intensity_3d(ct_file, point_location):
    x = point_location[0]
    y = point_location[1]
    z = point_location[2]
    img = ct_file.scan[z].pixel_array
    HU = img * ct_file.slice_properties[-2] + ct_file.slice_properties[-1]
    intensity_value = HU[y, x]
    #print('intensity_value is', intensity_value)
    return intensity_value



if __name__ == '__main__':
    #SEEDS = []
    path_point_surface_matching = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\" \
                                  "CT tooth segmentation\\CASE 5277\\CT3\\"
    #path_point_surface_matching = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\HW-9898 Point-to-surface experiments\\CT scan\\drill_fxt_0.2\\"
    # path = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\FA-human scan\\'

    # read ct images and check each slice to define the depth boundary for tooth segmentation
    test = CTImg(path_point_surface_matching)
    exit()
    test.plot_ct_slices()

    # Define z_boundaries
    z_min = np.int(input("Enter the value of minimum z \n"))
    z_max = np.int(input("Enter the value of maximum z \n"))
    selected_slice = z_min
    print('z_min is', z_min)
    print('z_max is', z_max)

    # Prepare for matplot
    test.read_ct(selected_slice)

    # Define tooth segmentation boxes (box method)
    start_tooth_segmentation = True
    while start_tooth_segmentation is True:
        fig1 = plt.figure()
        test.select_box(fig1)
        # input_flag = True
        # while input_flag is True:
        #     val = input("Press 'p' to proceed or 'r' to repeat data collection procedure \n")
        #     if val == 'r':
        #         fig1 = plt.figure()
        #         test.select_box(fig1)
        #     elif val == 'p':
        #         print('selected box is', test.corner_points)
        #         val = np.int(input("Enter the tooth number"))
        #         write_box_corners = True
        #         while write_box_corners is True:
        #             if -1 < val < 33:
        #                 test.corner_list[val-1] = test.corner_points.copy()
        #                 write_box_corners = False
        #             else:
        #                 val = np.int(input("Wrong number was entered, please enter a number between 0-32"))
        #         input_flag = False
        #     else:
        #         print('Wrong input is provided \n')
        val = np.int(input("Enter the tooth number"))
        test.corner_list[val - 1] = test.corner_points.copy()

        print('corner list is', test.corner_list)

        ask_if_continue = True
        while ask_if_continue is True:
            val = int(input("Continue to segment another tooth? \n press '1' to continue, press '0' to stop \n"))
            if val == 1:
                print('continue to segment the next tooth')
                ask_if_continue = False
            elif val == 0:
                print('segmentation is finished')
                start_tooth_segmentation = False
                ask_if_continue = False
            else:
                print('wrong value was entered \n')


    exit()
    pc = pcm.preprocess_point_cloud(region_candidate.boundary_final, voxel_size=4.0)
    print('shape of pc is', np.shape(pc))
    # source = o3d.PointCloud()
    # source.points = o3d.Vector3dVector(region_candidate.boundary_final)
    # source.paint_uniform_color([0, 0.651, 0.929])  # yellow
    # o3d.draw_geometries([source])

    plt.figure()
    ax = plt.axes(projection='3d')
    print('shape of boundary is', np.shape(region_candidate.boundary_final))
    Yomiplot.plot_3d_points(pc, ax, color='green', alpha=0.3, axe_option=False)

    plt.show()
