import plot as Yomiplot
# import open3d
from open3d import *
import open3d

import cv2
import geometry
import Readers as Yomiread


tooth_number = 10
surface = ['occlusal', 'lingual', 'buccal', 'front', 'back']

dicom_pc_file = 'C:\\tooth segmentation raw\\tooth_' + np.str(tooth_number) +'_surface_' + \
                surface[3] + '.csv'
points = Yomiread.read_csv(dicom_pc_file, 3)

pcd = open3d.PointCloud()
pcd.points = open3d.Vector3dVector(points)
visualization.draw_geometries([pcd])