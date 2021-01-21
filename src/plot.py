import Readers as Yomiread
import Kinematics as Yomikin
import numpy as np
import timeit
import scipy
import jstyleson
from scipy.optimize import minimize, rosen, rosen_der
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# Introduction on use
# 1. plot_kdl(joint, kdl, fig, arm_type_str)
#    plot forward kinematics
#    -args:
#           joint: joint angles one pose or multiple poses
#           kdl
#           fig
#           arm_type_str: a string to indicate the type of arms
#                         'GA' - current schunk 
#                         'UR' - UR arms
#                         'TA' - Tracker
#   
# 2. plot_kdl_df(joint,kdl,df_kdl,fig, arm_type_str)
#    plots FK and drill face nominal.
#    It works for both single pose and multiple poses.
#    -args:
#           joint: joint angles one pose or multiple poses
#           kdl
#           df_kdl: The drill face kdl. A 1x6 1-D array defined in Yomi convention X,Y,Z,Rz,Ry,Rx
#           fig
#           arm_type_str: a string to indicate the type of arms
#                         'GA' - current schunk 
#                         'UR' - UR arms
#                         'TA' - Tracker
#
#
# Notes:
# In case of re-scaling the axes, modify the corresponding values in the function.
# The FK is solved by Yomi convention. Convert models to yomi convention before use.



def plot_fk(joint, kdl, fig, arm_type_str):
    joint_tem0 = np.copy(joint)
    kdl_tem = np.copy(kdl)
    ax = fig.add_subplot(111, projection='3d')
    if np.ndim(joint_tem0) == 1:
        plot_fk_single_pose(joint_tem0, kdl_tem, ax, arm_type_str)
    else:
        for m in range(np.shape(joint_tem0)[0]):
            joint_tem = joint_tem0[m,:]
            plot_fk_single_pose(joint_tem, kdl_tem, ax, arm_type_str)
    
    # Plot working volume and base volume.
    #plot_box(0.10, 0.20, -0.05, 0.05, -0.2, -0.1, ax)
    #plot_box(-0.05, 0.05, -0.05, 0.05, 0., 0.1,ax)
    # Set up the axes limits
    ax.axes.set_xlim3d(left=-0.4999999, right=0.4999999)
    ax.axes.set_ylim3d(bottom=-0.4999999, top=0.4999999)
    ax.axes.set_zlim3d(bottom=-0.4999999, top=0.4999999)

    # Create axes labels
    ax.set_xlabel('X(m)')
    ax.set_ylabel('Y(m)')
    ax.set_zlabel('Z(m)')
    # plot_box(0.10, 0.20, -0.05, 0.05, -0.2, -0.1, ax)


def plot_fk_df(joint,kdl,df_kdl,fig, arm_type_str, t):
    joint_tem0 = np.copy(joint)
    kdl_tem = np.copy(kdl)
    df = np.copy(df_kdl)
    ax = fig.add_subplot(111, projection='3d')
    if np.ndim(joint_tem0) == 1:
        plot_fk_df_single_pose(joint_tem0, kdl_tem, df, ax, arm_type_str)
    else:
        for m in range(np.shape(joint_tem0)[0]):
            joint_tem = joint_tem0[m,:]
            plot_fk_df_single_pose(joint_tem, kdl_tem, df, ax, arm_type_str)

    # Plot working volume and base volume.
    #plot_box(0.10, 0.20, -0.05, 0.05, -0.2, -0.1, ax)
    #plot_box(-0.05, 0.05, -0.05, 0.05, 0., 0.1,ax)
    ax.scatter(t[0], t[1], t[2], zdir='z', s=20, c='r',
               rasterized=True)
    # Set up the axes limits
    ax.axes.set_xlim3d(left=-0.3999999, right=0.3999999)
    ax.axes.set_ylim3d(bottom=-0.3999999, top=0.3999999)
    ax.axes.set_zlim3d(bottom=-0.3999999, top=0.3999999)

    # Create axes labels
    ax.set_xlabel('X(m)')
    ax.set_ylabel('Y(m)')
    ax.set_zlabel('Z(m)')
    # plot_box(0.10, 0.20, -0.05, 0.05, -0.2, -0.1, ax)


def plot_fk_single_pose(joint, kdl, ax, arm_type_str):
    # Prepare data
    joint_tem = np.copy(joint)
    kdl_tem = np.copy(kdl)
    # Perform Forward Kinematics
    pos = Yomikin.FW_Kinematics_Matrices(kdl_tem, joint_tem)
    ee_pos = np.zeros(3)
    a_pos = []
    if arm_type_str == 'GA' or arm_type_str == 'UR':
        mj = 7
    elif arm_type_str == 'TA':
        mj = 8
    #ax = fig.add_subplot(111, projection='3d')
    for m in range(mj):
        eepos_tem = np.matmul(pos[m], [0, 0, 0, 1])
        a_pos_tem = pos[m][0:3,0:3]
        ee_pos = np.vstack((ee_pos, eepos_tem[0:3]))
        a_pos.append(a_pos_tem)
    for j in range(mj):

        # plot links
        # The function takes arguments in the order of
        # [x_start, x_end], [y_start, y_end], zs=[z_start, z_end]
        ax.plot([ee_pos[j, 0], ee_pos[j + 1, 0]], [ee_pos[j, 1], ee_pos[j + 1, 1]],
                zs=[ee_pos[j, 2], ee_pos[j + 1, 2]], c='r')

        # plot joint coordinate frames
        # check https://matplotlib.org/3.1.0/gallery/mplot3d/quiver3d.html for use
        # x, y, z vectors are the 1st, 2nd, 3rd columns in the rotation matrix.
        # z is in black, y is in green, x is in blue
        ax.quiver(ee_pos[j+1,0], ee_pos[j+1,1], ee_pos[j+1,2], a_pos[j][0,0], a_pos[j][1,0], a_pos[j][2,0],
                               pivot='tail',length=0.05,arrow_length_ratio=0.2, color='blue')
        ax.quiver(ee_pos[j+1,0], ee_pos[j+1,1], ee_pos[j+1,2], a_pos[j][0,1], a_pos[j][1,1], a_pos[j][2,1],
                               pivot='tail',length=0.05,arrow_length_ratio=0.2, color='g')
        ax.quiver(ee_pos[j+1,0], ee_pos[j+1,1], ee_pos[j+1,2], a_pos[j][0,2], a_pos[j][1,2], a_pos[j][2,2],
                               pivot='tail',length=0.05,arrow_length_ratio=0.2, color='k')

        # plot joint origins
        ax.scatter(ee_pos[j+1,0], ee_pos[j+1,1], ee_pos[j+1,2], color='black')


def plot_fk_df_single_pose(joint, kdl, df_kdl, fig, arm_type_str):
    # Prepare data
    joint_tem = np.copy(joint)
    kdl_tem = np.copy(kdl)
    df = np.copy(df_kdl)
    # Perform Forward Kinematics
    pos = Yomikin.FW_Kinematics_Matrices(kdl_tem, joint_tem)
    ee_pos = np.zeros(3)
    a_pos = []
    ax = fig
    if arm_type_str == 'GA' or arm_type_str == 'UR':
        mj = 7
    elif arm_type_str == 'TA':
        mj = 8
    #ax = fig.add_subplot(111, projection='3d')
    for m in range(mj):
        eepos_tem = np.matmul(pos[m], [0, 0, 0, 1])
        a_pos_tem = pos[m][0:3,0:3]
        ee_pos = np.vstack((ee_pos, eepos_tem[0:3]))
        a_pos.append(a_pos_tem)
    for j in range(mj):

        # plot links
        # The function takes arguments in the order of
        # [x_start, x_end], [y_start, y_end], zs=[z_start, z_end]
        ax.plot([ee_pos[j, 0], ee_pos[j + 1, 0]], [ee_pos[j, 1], ee_pos[j + 1, 1]],
                zs=[ee_pos[j, 2], ee_pos[j + 1, 2]], c='r')
        # plot joint origins
        ax.scatter(ee_pos[j+1,0], ee_pos[j+1,1], ee_pos[j+1,2], color='black')
    ax.scatter(0.,0.,0.,color='black')
    print('eepos is', ee_pos)
    # In case of Yomisetting parameters
    #T_df = np.reshape(df,newshape=(4,4))
    # In case of Matteo info
    T_df = Yomikin.Yomi_Base_Matrix(df)

    T_df_cb = np.matmul(pos[-1],T_df)
    ax.plot([ee_pos[-1,0], T_df_cb[0,3]],[ee_pos[-1,1], T_df_cb[1,3]],zs=[ee_pos[-1,2], T_df_cb[2,3]], color='green')
    ax.quiver(T_df_cb[0,3], T_df_cb[1,3], T_df_cb[2,3], T_df_cb[0,0], T_df_cb[1,0], T_df_cb[2,0], length=0.05,arrow_length_ratio=0.2, color='blue')


def plot_fk_dh(joint, kdl, fig, arm_type_str):
    joint_tem0 = np.copy(joint)
    kdl_tem = np.copy(kdl)
    ax = fig.add_subplot(111, projection='3d')
    if np.ndim(joint_tem0) == 1:
        plot_fk_dh_single_pose(joint_tem0, kdl_tem, ax, arm_type_str)
    else:
        for m in range(np.shape(joint_tem0)[0]):
            joint_tem = joint_tem0[m, :]
            plot_fk_dh_single_pose(joint_tem, kdl_tem, ax, arm_type_str)

    # Plot working volume and base volume.
    # plot_box(0.10, 0.20, -0.05, 0.05, -0.2, -0.1, ax)
    # plot_box(-0.05, 0.05, -0.05, 0.05, 0., 0.1,ax)
    # Set up the axes limits
    ax.axes.set_xlim3d(left=-0.4999999, right=0.4999999)
    ax.axes.set_ylim3d(bottom=-0.4999999, top=0.4999999)
    ax.axes.set_zlim3d(bottom=-0.4999999, top=0.4999999)

    # Create axes labels
    ax.set_xlabel('X(m)')
    ax.set_ylabel('Y(m)')
    ax.set_zlabel('Z(m)')
    # plot_box(0.10, 0.20, -0.05, 0.05, -0.2, -0.1, ax)


def plot_fk_dh_single_pose(joint, kdl, ax, arm_type_str):
    # Prepare data
    joint_tem = np.copy(joint)
    kdl_tem = np.copy(kdl)
    # Perform Forward Kinematics
    pos = Yomikin.FW_Kinematics_DH_classical(kdl_tem, joint_tem)
    ee_pos = np.zeros(3)
    a_pos = []
    if arm_type_str == 'GA' or arm_type_str == 'UR':
        mj = 6
    elif arm_type_str == 'TA':
        mj = 7
    #ax = fig.add_subplot(111, projection='3d')
    for m in range(mj):
        eepos_tem = np.matmul(pos[m], [0, 0, 0, 1])
        a_pos_tem = pos[m][0:3,0:3]
        ee_pos = np.vstack((ee_pos, eepos_tem[0:3]))
        a_pos.append(a_pos_tem)
    for j in range(mj):

        # plot links
        # The function takes arguments in the order of
        # [x_start, x_end], [y_start, y_end], zs=[z_start, z_end]
        ax.plot([ee_pos[j, 0], ee_pos[j + 1, 0]], [ee_pos[j, 1], ee_pos[j + 1, 1]],
                zs=[ee_pos[j, 2], ee_pos[j + 1, 2]], c='r')

        # plot joint coordinate frames
        # check https://matplotlib.org/3.1.0/gallery/mplot3d/quiver3d.html for use
        # x, y, z vectors are the 1st, 2nd, 3rd columns in the rotation matrix.
        # z is in black, y is in green, x is in blue
        ax.quiver(ee_pos[j+1,0], ee_pos[j+1,1], ee_pos[j+1,2], a_pos[j][0,0], a_pos[j][1,0], a_pos[j][2,0],
                               pivot='tail',length=0.05,arrow_length_ratio=0.2, color='blue')
        ax.quiver(ee_pos[j+1,0], ee_pos[j+1,1], ee_pos[j+1,2], a_pos[j][0,1], a_pos[j][1,1], a_pos[j][2,1],
                               pivot='tail',length=0.05,arrow_length_ratio=0.2, color='g')
        ax.quiver(ee_pos[j+1,0], ee_pos[j+1,1], ee_pos[j+1,2], a_pos[j][0,2], a_pos[j][1,2], a_pos[j][2,2],
                               pivot='tail',length=0.05,arrow_length_ratio=0.2, color='k')

        # plot joint origins
        ax.scatter(ee_pos[j+1,0], ee_pos[j+1,1], ee_pos[j+1,2], color='black')


# Plot 3D Boxes
# Only works for rectangular boxes
def plot_box (xmin,xmax,ymin,ymax,zmin,zmax, fig_control):
    # Plot surface in 3d
    # Source code can be checked from https://matplotlib.org/3.1.0/gallery/mplot3d/surface3d.html
    # Details about the function from https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
    X = np.arange(xmin, xmax, (xmax-xmin)/50)
    Y = np.arange(ymin, ymax, (ymax-ymin)/50)
    Z = np.arange(zmin, zmax, (zmax-zmin)/50)

    z_x,z_y = np.meshgrid(X,Y)
    y_x,y_z = np.meshgrid(X,Z)
    x_y,x_z = np.meshgrid(Y,Z)

    top_z = 0. * z_x + 0. * z_y + zmax
    bottom_z = 0. * z_x + 0. * z_y + zmin
    left_y = 0. * y_x + 0. * y_z + ymin
    right_y = 0. * y_x + 0. * y_z + ymax
    front_x = 0. * x_y + 0. * x_z + xmax
    back_x = 0. * x_y + 0. * x_z + xmin
    # top surface
    fig_control.plot_surface(z_x,z_y,top_z, alpha = 0.5, color = 'blue')
    fig_control.plot_surface(z_x,z_y,bottom_z, alpha = 0.5, color = 'blue')
    fig_control.plot_surface(y_x,left_y,y_z, alpha = 0.5, color = 'blue')
    fig_control.plot_surface(y_x,right_y,y_z, alpha = 0.5, color = 'blue')
    fig_control.plot_surface(front_x,x_y,x_z, alpha = 0.5, color = 'blue')
    fig_control.plot_surface(back_x,x_y,x_z, alpha = 0.5, color = 'blue')


# Plot Radar chart with multiple data sets
# categories variable includes the labels
# data variable is a list of data that to be plotted
# data_str variable includes the title of each data set
# Reference: https://plotly.com/python/radar-chart/
def plot_radar_multi(categories, data, data_str):
    fig = go.Figure()
    for data_tem,title_tem in zip(data, data_str):
        fig.add_trace(go.Scatterpolar(
            r = data_tem,
            theta=categories,
            name= title_tem,
            fill='toself'
        ))
    fig.update_layout(
        polar = dict(
            radialaxis=dict(
                visible=True,
                range=[0,100]
            )),
        showlegend=False
    )
    fig.show()

# Modifed data order for tracker dimensional analysis purpose
def plot_radar_multi_modify(categories, data, data_str):
    fig = go.Figure()
    for data_tem,title_tem in zip(data, data_str):
        new_data_tem = data_tem[1:]
        new_data_tem[0] = data_tem[1]
        new_data_tem[1] = data_tem[0]
        new_data_tem = np.insert(new_data_tem, 0, np.sqrt(2)*new_data_tem[0]*new_data_tem[-1]/(new_data_tem[0] + new_data_tem[-1]))
        new_data_tem = np.insert(new_data_tem, 2, np.sqrt(2)*new_data_tem[1]*new_data_tem[2]/(new_data_tem[1] + new_data_tem[2]))
        new_data_tem = np.insert(new_data_tem, 4, np.sqrt(2) * new_data_tem[3] * new_data_tem[4] / (new_data_tem[3] + new_data_tem[4]))
        new_data_tem = np.insert(new_data_tem, 6, np.sqrt(2) * new_data_tem[5] * new_data_tem[6] / (new_data_tem[5] + new_data_tem[6]))

        fig.add_trace(go.Scatterpolar(
            r = new_data_tem,
            theta=categories,
            name= title_tem,
            fill='toself'
        ))

    threshold = np.array([17.67766953, 25, 17.67766953, 25, 17.67766953, 25, 17.67766953, 25])
    fig.add_trace(go.Scatterpolar(
        r=threshold,
        theta=categories,
        name='Threshold',
        fill='toself'))

    fig.update_layout(
        polar = dict(
            radialaxis=dict(
                visible=True,
                range=[0,100]
            )),
        showlegend=True
    )
    fig.show()

# Modifed data order for tracker dimensional analysis purpose
def plot_radar_single_modify(categories, data, data_str):
    fig = go.Figure()
    data_tem = data
    title_tem = data_str
    new_data_tem = data_tem[1:]
    new_data_tem[0] = data_tem[1]
    new_data_tem[1] = data_tem[0]
    new_data_tem = np.insert(new_data_tem, 0,
                                 np.sqrt(2) * new_data_tem[0] * new_data_tem[-1] / (new_data_tem[0] + new_data_tem[-1]))
    new_data_tem = np.insert(new_data_tem, 2,
                                 np.sqrt(2) * new_data_tem[1] * new_data_tem[2] / (new_data_tem[1] + new_data_tem[2]))
    new_data_tem = np.insert(new_data_tem, 4,
                                 np.sqrt(2) * new_data_tem[3] * new_data_tem[4] / (new_data_tem[3] + new_data_tem[4]))
    new_data_tem = np.insert(new_data_tem, 6,
                                 np.sqrt(2) * new_data_tem[5] * new_data_tem[6] / (new_data_tem[5] + new_data_tem[6]))

    fig.add_trace(go.Scatterpolar(
            r=new_data_tem,
            theta=categories,
            name=title_tem,
            fill='toself'
        ))

    threshold = np.array([17.67766953, 25, 17.67766953, 25, 17.67766953, 25, 17.67766953, 25])
    fig.add_trace(go.Scatterpolar(
        r=threshold,
        theta=categories,
        name='Threshold',
        fill='toself'))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True
    )
    fig.show()
