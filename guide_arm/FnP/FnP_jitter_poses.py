import numpy as np
from src.Python.Kinematics import Kinematics as Yomikin
from src.Python.GuideArm import GuideArm as GA
from src.Python.InverseKinematicSolver import inverseKinematicSolver as IK
import matplotlib.pyplot as plt
from src.Python.plot import plot


# write csv file with data (matrix)
def write_csv_matrix(file_name, data, fmt = '%.4f'):
    f = open(file_name, 'w')
    data = np.asarray(data)
    if len(np.shape(data)) > 1:
        for data_tem in np.asarray(data):
            np.savetxt(f, data_tem.reshape(1, data_tem.shape[0]), delimiter=",", fmt = fmt)
    else:
        np.savetxt(f, data.reshape(1, len(data)), delimiter=',', fmt = fmt)
    f.close()


if __name__ == '__main__':
    # FnP nominal KDL
    KP_0_FNP = np.array([-0.219, 0.0, 0.136, 0.0, 0.0, 0.0,
                    0.0, -0.202, 0.0, 0.0, 0.0, -np.pi / 2.0,
                    0.410, 0.0, 0.0, -np.pi / 2.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, -np.pi / 2.0, 0.0, -np.pi / 2.0,
                    0.0, -0.405, 0.0, np.pi, 0.0, -np.pi / 2.0,
                    0.0, 0.0, 0.0, np.pi, 0.0, -np.pi / 2.0,
                    0.0, 0.0, 0.085, 0.0, 0.0, 0.0])

    # Schunk nominal KDL
    KP_0_SCHUNK = np.array([0., 0.167, 0.267, np.pi/ 2, 0., 0.,
                    0., 0., 0., np.pi, -np.pi/ 2, 0,
                    0.35, 0., 0., 0., 0., np.pi,
                    0., 0., 0., 0., np.pi/ 2, 0.,
                    0.305, 0., 0., 0., -np.pi/2, 0.,
                    0., 0., 0., 0., np.pi/ 2, 0.,
                    0., 0., 0.120875, np.pi/2, 0., 0])

    # Schunk jitter test poses
    p_schunk = np.array([[0.58793, 0.27922, -1.66853, -0.7877, -1.64784, 0.72724],
                        [0.49545, 1.79998, -0.21207, -1.3465, -1.81774, 0.47881],
                        [-0.58791, 0.29999, -1.66857, 0.78777, -1.64785, -0.72997],
                        [-0.55587, 1.80001, -0.33414, 1.40462, -2.14658, -0.89494],
                        [0.5879, 0.29997, -1.66855, -0.78767, -1.64789, 0.72724],
                        [0.55336, 1.79999, -0.33219, -1.5266, 0.11643, 0.90617]])

    # FnP jitter tests poses guess
    p_FnP_guess = np.array([[0.58793, -1.27922, -1.66853, -0.7877, -1.64784, 0.72724],
                        [0.49545, 2.79998, -0.21207, -1.3465, -1.81774, 0.47881],
                        [-0.58791, -1.29999, -1.66857, 0.78777, -1.64785, -0.72997],
                        [-0.55587, 1.80001, -0.33414, 1.40462, -2.14658, -0.89494],
                        [0.5879, -1.29997, -1.66855, -0.78767, -1.64789, 0.72724],
                        [0.55336, 1.79999, -0.33219, -1.5266, 0.11643, 0.90617]])

    # Estimate ee using Schunk nominal KDL (desired position vector for FnP arm)
    p_desire = []
    for pose in p_schunk:
        ee = Yomikin.FW_Kinematics_Matrices(KP_0_SCHUNK, pose)[-1]
        ee_desire = Yomikin.get_components_standard(ee)
        p_desire.append(ee_desire)
    p_desire = np.asarray(p_desire)
    joint_angle_guess = np.copy(p_FnP_guess)
    t_sf_df = np.eye(4)

    # Perform IK to find FnP poses (p_FnP)
    p_FnP = []
    diff = []
    for i in range(p_schunk.shape[0]):
        solution, pfinal = IK.solve_ik(KP_0_FNP, joint_angle_guess[i], p_desire[i], t_sf_df, 'FnP')
        p_FnP.append(solution)
        diff.append(p_desire[i] - pfinal)

    # Write results and plot for preview.
    result_file = "G:\\My Drive\\Project\\FnP\\Jitter Test\\robot_slow_speed_jogging_poses_fnp.csv"
    write_csv_matrix(result_file, p_FnP, fmt="%.5f")

    for j in range(6):
        fig = plt.figure()
        plot.plot_fk(p_FnP[j], KP_0_FNP, fig, 'UR')
    fig = plt.figure()
    plot.plot_fk(p_FnP, KP_0_FNP, fig, 'UR')
    #plt.show()

    schunkArm = GA.GuideArm('Schunk')
    fnpARM = GA.GuideArm('F&P')

    schunkArm.t_sf_df_use = np.eye(4)
    fnpARM.t_sf_df_use = np.eye(4)

    fnpGravCalPoses = np.zeros((np.size(p_schunk, 0), np.size(p_schunk, 1)))
    returnBothSolutions = True

    for i in range(np.size(p_schunk, 0)):
        poseIn = Yomikin.forwardKinematics(schunkArm.robot.yomiParams, p_schunk[i, :], 'p_s')
        poses, poseOut = fnpARM.solve_IK(poseIn, returnBothSolutions)
        fnpGravCalPoses[i, :] = poses[1]

    print('fnpGravCal is')
    print(fnpGravCalPoses)
    print('diff is')
    print(fnpGravCalPoses - p_FnP)
    fig = plt.figure()
    plot.plot_fk(fnpGravCalPoses, KP_0_FNP, fig, 'UR')

    plt.show()

