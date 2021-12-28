import numpy as np
import matplotlib.pyplot as plt
import DICOM_reader as DR

# import slider bar in matplot
from matplotlib.widgets import Slider, Button


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


# # interact with matplot figure
# def onclick(event):
#     global ix, iy
#     ix, iy = event.xdata, event.ydata
#     print('x = %d, y = %d'%(ix, iy))
#     HU_new = np.copy(HU)
#
#     global coords, centers_pc, radius_pc
#     coords.append((np.int(ix), np.int(iy)))
#     if len(coords) % 2 == 0:
#         centers_pc.clear()
#         radius_pc.clear()
#         for i in range(len(coords)//2):
#             circle_points, radius = draw_circle(coords[i*2], coords[i*2 + 1])
#             centers_pc.append(coords[i*2])
#             radius_pc.append(radius)
#             for point in circle_points:
#                 HU_new[point[1], point[0]] = 3000
#     plt.imshow(HU_new)
#     fig1.canvas.draw()


# define CT_scan class
class interactive_seed_selection:
    def __init__(self, img_pixel_array):
        self.pixel_array = np.copy(img_pixel_array)
        self.pixel_array_tem = np.copy(self.pixel_array)
        self.click_points = []
        self.seed = []
        self.search_radius = []
        self.figure = None
        self.input_flag = True
        self.ix = 0
        self.iy = 0

    def select_seed(self, figure_control):
        self.seed.clear()
        self.search_radius.clear()
        self.click_points.clear()
        self.pixel_array_tem = np.copy(self.pixel_array)
        self.figure = figure_control
        plt.imshow(self.pixel_array)
        print('Select seed points and radius in figure')
        self.figure.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def onclick(self, event):
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
        self.figure.canvas.draw()

    

# read all ct images
def ct_reader(ct_file_path):
    scan, slice_properties = DR.load_scan_human(ct_file_path)
    init_slice = 35

    img = scan[init_slice].pixel_array
    HU = img * slice_properties[-2] + slice_properties[-1]

    fig, ax = plt.subplots()
    plt.imshow(HU)
    ax.set_xlabel('Slice')
    axcolor = 'lightgoldenrodyellow'
    ax.margins(x=0)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal oriented slider to control the slice
    axslice = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    slice_slider = Slider(
        ax=axslice,
        label="Slice",
        valmin=0,
        valmax=slice_properties[2],
        valinit=init_amplitude
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        # ax.clf()
        ax.clear()
        ax.imshow(scan[np.int(val)].pixel_array)
        # fig.canvas.draw()
        # fig.canvas.draw_idle()

    slice_slider.on_changed(update)
    plt.show()


SEEDS = []
path_point_surface_matching = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\CT tooth segmentation\\CASE 5277\\CT3\\"
#path_point_surface_matching = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\HW-9898 Point-to-surface experiments\\CT scan\\drill_fxt_0.2\\"
# path = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\FA-human scan\\'

slice_number = 265
scan, slice_properties = DR.load_scan_human(path_point_surface_matching)
img = scan[slice_number].pixel_array
HU = img * slice_properties[-2] + slice_properties[-1]

init_amplitude = 35

fig, ax = plt.subplots()
plt.imshow(HU)
#ax.set_xlabel('Time [s]')

axcolor = 'lightgoldenrodyellow'
ax.margins(x=0)
plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a vertically oriented slider to control the amplitude
axamp = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
#axamp = plt.axes(facecolor=axcolor)
amp_slider = Slider(
    ax=axamp,
    label="Amplitude",
    valmin=0,
    valmax=slice_properties[2],
    valinit=init_amplitude
    #orientation="vertical"
)

# The function to be called anytime a slider's value changes
def update(val):
    #ax.clf()
    ax.clear()
    ax.imshow(scan[np.int(val)].pixel_array)
    #fig.canvas.draw()
    #fig.canvas.draw_idle()

amp_slider.on_changed(update)
plt.show()

exit()




#--- interactive plot ---#
fig1 = plt.figure()
test_reader = interactive_seed_selection(HU)
test_reader.select_seed(fig1)

input_flag = True
while input_flag is True:
    val = input("Press 'p' to proceed or 'r' to repeat data collection procedure \n")
    if val == 'r':
        fig1 = plt.figure()
        print('shape of center pc is', len(test_reader.seed))
        test_reader.select_seed(fig1)
    elif val == 'p':
        print('shape of center pc is', len(test_reader.seed))
        input_flag = False
    else:
        print('Wrong input is provided \n')

exit()

scan, slice_properties = DR.load_scan_human(path_point_surface_matching)
img = scan[265].pixel_array

HU = img * slice_properties[-2] + slice_properties[-1]

coords = []
centers_pc = []
radius_pc = []

fig1 = plt.figure()
plt.imshow(HU)
print('Select seed points and radius in figure')
cid = fig1.canvas.mpl_connect('button_press_event', onclick)
plt.show()

input_flag = True
while input_flag is True:
    val = input("Press 'p' to proceed or 'r' to repeat data collection procedure")
    if val == 'r':
        coords.clear()
        centers_pc.clear()
        radius_pc.clear()
        fig1 = plt.figure()
        plt.imshow(HU)
        print('Select seed points and radius in figure')
        cid = fig1.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
    elif val == 'p':
        print('shape of center pc is', len(centers_pc))
        input_flag = False
    else:
        print('Wrong input is provided')