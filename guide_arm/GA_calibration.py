import Kinematics as Yomikin
import numpy as np
import scipy
from scipy.optimize import minimize, rosen, rosen_der
import os


# write csv file with data
def write_csv(file_name, data, fmt='%.4f'):
    np.savetxt(file_name, data, delimiter=",", newline="\n", fmt = fmt)


def unpack(kp_2b_update, p, kp_update_parameters):
    # the np.copy() used here is to solve a known issue regarding pointers. please leave as it for now.
    kp = np.copy(kp_2b_update)
    phat = np.copy(p)
    kp_fit = kp_update_parameters[:]
    xx = np.nonzero(kp_fit)
    kp[xx] = phat
    k = kp[:]
    return k


def pack(kp_2b_packed, kp_update_parameters):
    # the np.copy() used here is to solve a known issue regarding pointers. please leave as it for now.
    kp = np.copy(kp_2b_packed)
    kp_p = np.copy(kp_update_parameters)
    p_tem = kp[np.nonzero(kp_p)]
    p = np.array(p_tem)
    return p


def looperror(cal, gaq, gaf):
    kp_tem = np.copy(KP_0)
    cal_tem = np.copy(cal)
    kp_fit_tem = np.copy(KP_FIT)

    cal_tem = unpack(kp_tem, cal_tem, kp_fit_tem)
    err = np.zeros(shape=(3*gaq.shape[0],))
    err_norm = np.zeros(shape=(gaq.shape[0],))
    position_error = np.zeros(shape=(gaq.shape[0], 3))
    for x in range(gaq.shape[0]):
        joint_tem_f = np.insert(gaq[x, :], 0, 0.)
        eepos = Yomikin.FW_Kinematics_Matrices_no_common(cal_tem, joint_tem_f)
        # If the common reference is removed, use '6'
        # If the common reference is kept, use '7'
        eepos = np.matmul(eepos[-1], [0, 0, 0, 1])
        position_error[x,:] = eepos[0:3] - gaf[x,:]
        err[3 * x + 0,] = position_error[x, 0]
        err[3 * x + 1,] = position_error[x, 1]
        err[3 * x + 2,] = position_error[x, 2]
        err_norm[x,] = np.linalg.norm(eepos[0:3] - gaf[x,:])

    rms = np.sqrt(np.mean(err_norm**2))
    variance = np.sum((err - np.mean(err)) **2)/err.shape[0]
    stdev = np.sqrt(variance)
    print('Mean residual distance error is', np.mean(err)*1000, 'mm')
    print('Standard deviation of residual distance is', stdev*1000, 'mm')
    print('Calibration rms is', rms * 1000, 'mm')
    return err


def opt(gaq, gaf):
    kp_tem = np.copy(KP_0)
    kp_fit_tem = np.copy(KP_FIT)
    opt = pack(kp_tem, kp_fit_tem)
    p, cov, infodict, mesg, ier = scipy.optimize.leastsq(looperror, opt,
                                                         (gaq, gaf),
                                                         ftol=1e-10, xtol=1e-7,
                                                         full_output=True)
    print('integer flag is', ier)
    print('message is', mesg)
    jac = infodict["fjac"]
    # the conditional number calculated this way is different from the Neok value.
    jacobian_condition_number = np.linalg.cond(jac)
    print('Calibration condition number is', jacobian_condition_number)
    tem_p = p[:]
    kdl = tem_p[:]
    print('kdl is', kdl)
    kdl_unpack = unpack(kp_tem, kdl, kp_fit_tem)
    print('kdl_unpack', kdl_unpack)
    return kdl_unpack


if __name__ == '__main__':
    # FnP nominal KDL
    KP_0 = np.array([-0.219, 0.0, 0.136, 0.0, 0.0, 0.0,
                    0.0, -0.202, 0.0, 0.0, 0.0, -np.pi / 2.0,
                    0.410, 0.0, 0.0, -np.pi / 2.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, -np.pi / 2.0, 0.0, -np.pi / 2.0,
                    0.0, -0.405, 0.0, np.pi, 0.0, -np.pi / 2.0,
                    0.0, 0.0, 0.0, np.pi, 0.0, -np.pi / 2.0,
                    0.0, 0.0, 0.085, 0.0, 0.0, 0.0])

    # FnP fitting parameters
    KP_FIT = np.transpose(np.array([1, 1, 0, 0, 1, 1,  # Common link
                                    1, 1, 0, 1, 0, 1,  # Link1
                                    1, 1, 0, 0, 1, 1,
                                    1, 1, 0, 1, 0, 1,
                                    1, 1, 0, 1, 0, 1,
                                    1, 1, 0, 1, 0, 1,
                                    1, 1, 1, 0, 0, 0]))

    # Prepare data
    folder = 'G:/My Drive/Project/FnP/GA calibration/Trial1/robot-kincal-data/'
    for file in os.listdir(folder):
        if file == 'robot_points.txt':
            GAq = np.genfromtxt(folder + file, delimiter=',')[:,0:6]
        if file == 'faro_points.txt':
            GAfiducial = np.genfromtxt(folder + file, delimiter=',')[:,2:] / 1000

    if any(v is None for v in [GAq, GAfiducial]):
        print('Error, missing data')
        print('Make sure robot_points and faro_points files present in the folder')
        raise RuntimeError

    kdl = opt(GAq, GAfiducial)
    result_file = folder + "/Yomisettings_debug.csv"
    write_csv(result_file, kdl, fmt="%.8f")





