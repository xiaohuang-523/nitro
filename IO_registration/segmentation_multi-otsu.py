# Splint fitting for full dental arch
# Original Code:  SilverRhino/echairez/getSplinePoints/src/SplineFinderEC.cpp
#
# Author: Xiao Huang, Data: 02/23/2021

import cv2
import numpy as np
from matplotlib import pyplot as plt

from skimage import data, io, img_as_ubyte, measure
from skimage.color import label2rgb, rgb2gray
from skimage.filters import threshold_multiotsu
# import libraries for edge detection
from skimage.filters import roberts, sobel, scharr, prewitt, farid

from scipy.spatial.kdtree import KDTree

# import user define library
import array_processing as ap
import image_reader

#   Function Description
#   Scale a numpy matrix
def scale(col, min, max):
    range = col.max() - col.min()
    a = (col - col.min()) / range
    return a * (max - min) + min


#   Function Description
#   Convert an image to grayscale (0,255)
def convert_to_grayscale(image):
    print('converting image to grayscale')
    # Convert pixel_array (img) to -> gray image (img_2d_scaled)
    # Step 1. Convert to float to avoid overflow or underflow losses.
    print('  convert datatype to float')
    img_2d = image.astype(float)
    # Step 2. Rescaling grey scale between 0-255
    print('  scale to grayscale')
    img_2d_scaled = scale(img_2d, 0, 255)
    # Step 3. Convert to uint
    print('  convert datatype to unit8')
    img_2d_scaled = np.uint8(img_2d_scaled)
    return img_2d_scaled


#   Function Description
#   Apply multiple otsu filter
#   tutorial: https://www.youtube.com/watch?v=YdhhiXDQDl4&list=PLZsOBAyNTZwbIjGnolFydAN33gyyGP7lT&index=112
#   Package tutorial
def multiOtsu(th_number, image):
    thresholds = threshold_multiotsu(image, classes=th_number+1)
    regions = np.digitize(image, bins=thresholds)
    output = img_as_ubyte(regions)
    return output


#   Function Description
#   Image preprocessing using filters
#       1. Apply multi otsu filters and m number of thresholds. (currently m = 4)
#       2. Segment the images based on pixel thresholds defined in step1.
#       3. Apply binary otsu filter
#       4. Apply errosion and dilation to smooth the image. (optional for DICOM)
#       5. Identify and label different regions in the image.
#       6. Perform Edge detection
#   tutorial: https://www.youtube.com/watch?v=YdhhiXDQDl4&list=PLZsOBAyNTZwbIjGnolFydAN33gyyGP7lT&index=112
#   Package tutorial
#
#   Input: grayscale image.
#   Output: Binary threshold image, labeled image, edge detection
def image_pre_process(image):
    # Apply otsu filter to find the thresholds
    print('starting with multi otsu filter')
    thresholds = threshold_multiotsu(image, classes=THO_N + 1)
    print('thresholds are', thresholds)
    regions = np.digitize(image, bins=thresholds)
    otsu_regions_img = img_as_ubyte(regions)

    # Binary threshold methods
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
    ret, img_binary = cv2.threshold(image, thresholds[0], 255, cv2.THRESH_BINARY)

    # Dilation and erosion
    # https://www.youtube.com/watch?v=WQK_oOWW5Zo
    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(img_binary, kernel, iterations=1)
    dilation = cv2.dilate(img_binary, kernel, iterations=1)
    opening = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)

    # Label image
    # https://www.youtube.com/watch?v=u3nG5_EjfM0&list=PLZsOBAyNTZwbIjGnolFydAN33gyyGP7lT&index=119
    #label_image = measure.label(otsu_regions_img, connectivity=img_binary.ndim)
    label_image = measure.label(opening, connectivity=img_binary.ndim)
    image_label_overlay = label2rgb(label_image, image=otsu_regions_img)


    # Get different regions from the labeled image
    # https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html
    i = 0
    regions = []
    bbx = []
    for region in measure.regionprops(label_image):
        i += 1
        print('i is', i)
        print('region is', region.image)
        regions.append(region.image)
        bbx.append(region.bbox)
        figure = plt.figure()
        plt.imshow(region.image)
        plt.title('region ' + np.str(i))
        plt.savefig('G:\\My Drive\\Project\\IntraOral Scanner Registration\\Results\\Segmentation test\\region ' + np.str(i))
        plt.close()
    print('shape of regions is', np.shape(regions))
    for bbx_ind in bbx:
    #figure = plt.figure()
    #plt.imshow(regions[-3])
    #plt.title('label image check')
        print('region bounding box is', bbx_ind)

    # Edge detection
    # https://www.youtube.com/watch?v=Oy4duAOGdWQ&list=PLZsOBAyNTZwbIjGnolFydAN33gyyGP7lT&index=105
    robert_image = roberts(img_binary)
    sobel_image = sobel(img_binary)
    scharr_image = scharr(img_binary)
    prewitt_image = prewitt(img_binary)
    farid_image = farid(img_binary)
    print('detected edge is', sobel_image)

    return otsu_regions_img, image_label_overlay, sobel_image



#   Function Description
#   Find local maxima or minima in 3d point cloud
#   tutorial: https://stackoverflow.com/questions/27032562/local-maxima-in-a-point-cloud
#   Package tutorial
def locally_extreme_points(coords, data, neighbourhood, lookfor = 'max', p_norm = 2.):
    # Description
    # Find local maxima of points in a pointcloud.  Ties result in both points passing through the filter.
    #
    # Not to be used for high-dimensional data.  It will be slow.
    #
    # coords: A shape (n_points, n_dims) array of point locations
    # data: A shape (n_points, ) vector of point values
    # neighbourhood: The (scalar) size of the neighbourhood in which to search.
    # lookfor: Either 'max', or 'min', depending on whether you want local maxima or minima
    # p_norm: The p-norm to use for measuring distance (e.g. 1=Manhattan, 2=Euclidian)
    #
    # returns
    #     filtered_coords: The coordinates of locally extreme points
    #     filtered_data: The values of these points

    assert coords.shape[0] == data.shape[0], 'You must have one coordinate per data point'
    extreme_fcn = {'min': np.min, 'max': np.max}[lookfor]
    kdtree = KDTree(coords)
    neighbours = kdtree.query_ball_tree(kdtree, r=neighbourhood, p = p_norm)
    i_am_extreme = [data[i]==extreme_fcn(data[n]) for i, n in enumerate(neighbours)]
    extrema, = np.nonzero(i_am_extreme)  # This line just saves time on indexing
    return coords[extrema], data[extrema]


if __name__ == '__main__':
    # define global paramters
    THO_N = 4 # 4 for RD-08-L

    # read dicom file
    path = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\RD-08-L\\'
    path_full_arch = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\DICOMs_Typodont\\full_arch_voxel_02\\'
    #path = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\FA-human scan\\'
    scan, slice_properties = image_reader.load_scan_human(path_full_arch)
    img = scan[172].pixel_array # 170 and 165 for RD-08-L   # 290 for human  #

    # convert to grayscale for image processing
    img_gray = convert_to_grayscale(img)

    # flip dicom image along its vertical axis
    #img_gray = cv2.flip(img_gray, 1)   # flip along image's vertical axis
    #flippedimage = cv2.flip(image, 0)   # flip along image's horizontal axis
    #flippedimage = cv2.flip(image, -1)  # flip along horizontal and vertical axes

    im = image_pre_process(img_gray)

    # plot histogram
    plt.figure()
    plt.hist(img_gray.ravel(), bins=100, range=(0,255))
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylim((0, 5000))
    plt.ylabel("pixels")

    # Apply ostu filter to find the thresholds
    print('starting with multi otsu filter')
    thresholds = threshold_multiotsu(img_gray, classes=5)
    print('thresholds are', thresholds)
    regions = np.digitize(img_gray, bins=thresholds)
    output = img_as_ubyte(regions)

    # Segment based on pixel thresholds
    img_extract = ap.extract_array_elements(img_gray, thresholds[1], 0, -1)
    img_segment_idx = img_extract[0]
    img_segment_element = img_extract[1]
    img_segment = img_extract[2]

    # Binary threshold methods
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
    ret, img_binary = cv2.threshold(img_segment, thresholds[0], 255, cv2.THRESH_BINARY)

    # Dilation and erosion
    # https://www.youtube.com/watch?v=WQK_oOWW5Zo
    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(img_binary, kernel, iterations=1)
    dilation = cv2.dilate(img_binary, kernel, iterations=1)

    # Label image
    # https://www.youtube.com/watch?v=u3nG5_EjfM0&list=PLZsOBAyNTZwbIjGnolFydAN33gyyGP7lT&index=119
    label_image = measure.label(output, connectivity=img_binary.ndim)
    image_label_overlay = label2rgb(label_image, image=output)

    # Edge detection
    # https://www.youtube.com/watch?v=Oy4duAOGdWQ&list=PLZsOBAyNTZwbIjGnolFydAN33gyyGP7lT&index=105
    robert_image = roberts(img_binary)
    sobel_image = sobel(img_binary)
    scharr_image = scharr(img_binary)
    prewitt_image = prewitt(img_binary)
    farid_image = farid(img_binary)



    #img_mutso = multiOtsu(3, img)
    plt.figure()
    plt.imshow(img_gray, cmap='gray')
    plt.title("Grayscale original image")

    plt.figure()
    plt.imshow(output, cmap='gray')
    plt.title("Ostu image")

    plt.figure()
    plt.imshow(img_segment, cmap='gray')
    plt.title("Segment image")

    plt.figure()
    plt.imshow(img_binary, cmap='gray')
    plt.title("Binary threshold image")

    plt.figure()
    plt.imshow(erosion, cmap='gray')
    plt.title("Erosion image")

    plt.figure()
    plt.imshow(dilation, cmap='gray')
    plt.title("Dilation image")

    plt.figure()
    plt.imshow(image_label_overlay)
    plt.title("Label image")

    # robert_image = roberts(img_binary)
    # sobel_image = sobel(img_binary)
    # scharr_image = scharr(img_binary)
    # prewitt_image = prewitt(img_binary)
    # farid_image = farid(img_binary)

    plt.figure()
    plt.imshow(robert_image)
    plt.title("Robert image")

    plt.figure()
    plt.imshow(im[0])
    plt.title("Function test otsu image")

    plt.figure()
    plt.imshow(im[1])
    plt.title("Function test label image")

    plt.figure()
    plt.imshow(im[2])
    plt.title("Function test edge image")

    #plt.imshow(img_mutso)
    plt.show()
