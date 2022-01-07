import numpy as np
import Readers as Yomiread
import Writers as Yomiwrite

def transpose_pc(pc_2b_convert, transformation):
    pc_converted = []
    if pc_2b_convert.ndim == 1:
        tem = pc_2b_convert
        tem = np.insert(tem, 3, 1.)
        tem_converted = np.matmul(transformation, tem)[0:3]
        pc_converted = tem_converted
    else:
        for point in pc_2b_convert:
            tem = np.insert(point, 3, 1.)
            tem_converted = np.matmul(transformation, tem)[0:3]
            pc_converted.append(tem_converted)
    return np.asarray(pc_converted)

mtx_file = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\TRE_verification\\result\\transforamtion_from_fiducial_space_to_ct.csv"
mtx_f2c = Yomiread.read_csv(mtx_file, 4, 5, flag=1)

splint_file = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\Yomiplan fiducial registration\\Splint files\\FXT-0086-07-LRUL-MFG-Splint.txt"

fiducial_fiducial_space = Yomiread.read_csv(splint_file, 4, 9, flag=1, delimiter=" ")[:, 1:4]
print('fiducial is', fiducial_fiducial_space)
fiducial_ct_space = transpose_pc(fiducial_fiducial_space, mtx_f2c)

python_yomi = np.eye(4)
python_yomi[0:3, 3] = np.array([0, 0, (304+130)*0.2])
python_yomi[0:3, 0] = np.array([0, 1, 0])
python_yomi[0:3, 1] = np.array([1, 0, 0])
python_yomi[0:3, 2] = np.array([0, 0, -1])

python_yomi_real = np.linalg.inv(python_yomi)
print('fiducial_ct_space is', fiducial_ct_space)

fiducial_ct_space = transpose_pc(fiducial_ct_space, python_yomi_real)
print('fiducial_ct_space is', fiducial_ct_space)

RESULT_TEM_BASE = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\TRE_verification\\result\\"
Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + 'fiducials_in_ct.csv', fiducial_ct_space,
                           fmt='%0.8f')

