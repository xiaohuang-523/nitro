import jstyleson
import numpy as np
import pandas as pd
import csv


# Read Guide arm parameters from calibration file
def read_YomiSettings_schunk_kdl(fname):
    json_file=open(fname)
    data=jstyleson.load(json_file)
    GACal=np.asarray(data['schunk_kdl'])
    return GACal


# Read Tracker arm parameters from calibration file
def read_YomiSettings_tracker_kdl(fname):
    json_file=open(fname)
    data=jstyleson.load(json_file)
    TACal=np.asarray(data['tracker_kdl'])
    return TACal


def read_YomiSettings(fname, str):
    json_file=open(fname)
    data=jstyleson.load(json_file)
    TACal=np.asarray(data[str])
    return TACal


# Read the ball bar calibration measurements, *.m files
def read_calibration_measurements(fname):
    file = open(fname, "r")
    data = file.readlines()
    file.close()
    trigger = 0
    bb_data = []
    for i in range(len(data)):
        if "data" in data[i]:
            startIndex = i
            trigger = 1
        elif "];\n" in data[i]:
            endIndex = i-1
            if trigger == 1:
                break
        if trigger == 1:
            if i > startIndex:
                bb_data.append(data[i])
    for i in range(endIndex-startIndex-1):
        bb_data[i] = bb_data[i][0:-2]
    bb_data[endIndex-startIndex-1] = bb_data[endIndex-startIndex-1][0:-2]
    lc = len(bb_data)
    bb_tem = np.zeros(16)
    for j in range(lc):
        bb_data_t = np.fromstring(bb_data[j], dtype=float, sep=',')
        bb_tem = np.vstack((bb_tem, bb_data_t))
    bb_data_f = bb_tem[1:,:]
    return bb_data_f


# Read the ball measurements text file for ball position and bar ball length
def read_ball_measurements(fname):
    bbdata=pd.read_csv(fname, header = None)
    ball1=np.asarray([bbdata[2][3],bbdata[3][3],bbdata[4][3]])
    ball2=np.asarray([bbdata[2][4],bbdata[3][4],bbdata[4][4]])
    ball3=np.asarray([bbdata[2][5],bbdata[3][5],bbdata[4][5]])
    bar1=float(bbdata[1][6]) ; bar2=float(bbdata[1][7]) ; bar3=float(bbdata[1][8])
    output=(ball1,ball2,ball3,bar1,bar2,bar3)
    print(output)
    return ball1, ball2, ball3, bar1, bar2, bar3


# Parse excel files. Information: https://www.sitepoint.com/using-python-parse-spreadsheet-data/
# def read_xls(fname):
#     workbook = xlrd.open_workbook(fname)
#     worksheet = workbook.sheet_by_index(0)
#     trigger = 1
#     i = 0
#     bb_data = []
#     bb_tem = np.zeros(7)
#     if trigger == 1:
#         for j in range(7):
#             bb_data = np.append(bb_data, worksheet.cell(i,j).value)
#         i += 1
#         if worksheet.cell(i, 0).value == xlrd.empty_cell.value:
#             row = i
#             trigger = 0
#     bb_data_f = np.reshape(bb_data, (row, 7))
#     return bb_data_f

# Parse csv files. Information: https://docs.python.org/3/library/csv.html
def read_csv(fname, jnumber = 7, line_number = 50000):
    bb_tem = np.zeros(jnumber)
    bb_data_t = np.zeros(jnumber)
    flag = 0
    count = 0
    with open(fname, newline='') as f:
    #with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        #print('reader is', reader)
        for row in reader:
            count += 1
            if flag == 0:
                flag = 1
            elif flag == 1:
                for j in range(jnumber):
                    bb_data_t[j] = np.fromstring(row[j], dtype=float, sep=',')
                bb_tem = np.vstack((bb_tem, bb_data_t))
            if count == line_number:
                break
    bb_data_f = bb_tem[1:, :]
    return bb_data_f


def read_csv_simple(fname):
    data = []
    with open(fname, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data_tem = np.fromstring(row[0],dtype=float, sep=',')
            data = np.append(data, data_tem)
    return data


def read_csv_multiple_column(fname, column_number):
    data = np.zeros(column_number)
    row_data = np.zeros(column_number)
    with open(fname, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            for i in range(column_number):
                print('row is ', row)
                row_data[i] = np.fromstring(row[i], dtype=float, sep=',')
            data = np.vstack((data,row_data))
    return np.asarray(data[1:,:])


def read_txt(fname):
    data= []
    txt_file = open(fname,'r')
    FILE = txt_file.readlines()
    for line in range(len(FILE)):
        data_tem = np.fromstring(FILE[line],dtype=float,sep=',')
        data.append(data_tem)
    return np.asarray(data)

# Read log file
def readLog_JointAngle(log_file_path, startLine, endLine):
    log_file = open(log_file_path, 'r')
    FILE = log_file.readlines()
    QRA_time_list = []
    QRA_list = []
    QRD_time_list = []
    QRD_list = []
    QRA = np.zeros(6)
    QRD = np.zeros(6)
    QRA_time = []
    QRD_time = []
    # use the last line if endLine = -1
    if endLine == -1:
        endLine = len(FILE) - 100
    else:
        endLine = endLine

    for line in range(startLine,endLine):
        if "QRA" in FILE[line]:
            QRA_list.append(FILE[line][65:])
            QRA_time_list.append(FILE[line][11:24])
        if "QRD" in FILE[line]:
            QRD_list.append(FILE[line][65:])
            QRD_time_list.append(FILE[line][11:24])

    for j in range(len(QRA_list)):
        QRA_data_t = np.fromstring(QRA_list[j], dtype=float, sep=',')
        QRA = np.vstack((QRA, QRA_data_t))
    QRA = QRA[1:,:]

    for k in range(len(QRD_list)):
        QRD_data_t = np.fromstring(QRD_list[k], dtype=float, sep=',')
        QRD = np.vstack((QRD, QRD_data_t))
    QRD = QRD[1:, :]

    for l in range(len(QRA_time_list)):
        h, m, s = QRA_time_list[l].split(':')
        QRA_time.append(int(h) * 3600 + int(m) * 60 + float(s))

    for m in range(len(QRD_time_list)):
        h, m, s = QRD_time_list[m].split(':')
        QRD_time.append(int(h) * 3600 + int(m) * 60 + float(s))

    return QRA, QRA_time, QRD, QRD_time


def readLog_JointAngle_FnP(log_file_path, startLine, endLine, UDP_array_length = 20):
    log_file = open(log_file_path, 'r')
    FILE = log_file.readlines()
    QRA_time_list = []
    QRA_list = []
    QRD_time_list = []
    QRD_list = []
    QRA = np.zeros(6)
    QRD = np.zeros(6)
    QRA_time = []
    QRD_time = []

    UDPA_time_list = []
    UDPA_list = []
    UDPA = np.zeros(UDP_array_length)

    UDPD_time_list = []
    UDPD_list = []
    UDPD = np.zeros(8)

    # use the last line if endLine = -1
    if endLine == -1:
        endLine = len(FILE)
    else:
        endLine = endLine

    for line in range(startLine,endLine):
        if "QRA" in FILE[line]:
            QRA_list.append(FILE[line][55:])
            QRA_time_list.append(FILE[line][11:24])
        if "QRD" in FILE[line]:
            QRD_list.append(FILE[line][55:])
            QRD_time_list.append(FILE[line][11:24])
        if "received UDP response" in FILE[line]:
            UDPA_list.append(FILE[line][72:])
            UDPA_time_list.append(FILE[line][11:24])
        #if "preparing UDP command" in FILE[line]:
        if "sending UDP command" in FILE[line]:
            UDPD_list.append(FILE[line][67:])
            UDPD_time_list.append(FILE[line][11:24])


    for j in range(len(QRA_list)):
        QRA_data_t = np.fromstring(QRA_list[j], dtype=float, sep=',')
        QRA = np.vstack((QRA, QRA_data_t))
    QRA = QRA[1:,:]

    for k in range(len(QRD_list)):
        QRD_data_t = np.fromstring(QRD_list[k], dtype=float, sep=',')
        QRD = np.vstack((QRD, QRD_data_t))
    QRD = QRD[1:, :]

    for l in range(len(QRA_time_list)):
        h, m, s = QRA_time_list[l].split(':')
        QRA_time.append(int(h) * 3600 + int(m) * 60 + float(s))

    for m in range(len(QRD_time_list)):
        h, m, s = QRD_time_list[m].split(':')
        QRD_time.append(int(h) * 3600 + int(m) * 60 + float(s))

    for i in range(len(UDPA_list)):
        UDPA_data_t = np.fromstring(UDPA_list[i], dtype=float, sep=',')
        UDPA = np.vstack((UDPA, UDPA_data_t))
    UDPA = UDPA[1:, :]

    for i in range(len(UDPD_list)):
        UDPD_data_t = np.fromstring(UDPD_list[i], dtype=float, sep=',')
        UDPD = np.vstack((UDPD, UDPD_data_t))
    UDPD = UDPD[1:, :]
    return QRA, QRA_time, QRD, QRD_time, UDPA, UDPD


# Read log file
def readLog_JointAngle_with_idx(log_file_path, startLine, endLine):
    log_file = open(log_file_path, 'r')
    FILE = log_file.readlines()
    QRA_time_list = []
    QRA_idx_list = []
    QRA_list = []
    QRD_time_list = []
    QRD_idx_list = []
    QRD_list = []
    QRA = np.zeros(6)
    QRD = np.zeros(6)
    QRA_time = []
    QRD_time = []
    QRA_idx = []
    QRD_idx = []
    for line in range(startLine,endLine):
        if "QRA" in FILE[line]:
            QRA_list.append(FILE[line][65:])
            QRA_time_list.append(FILE[line][11:24])
            QRA_idx.append(line)
        if "QRD" in FILE[line]:
            QRD_list.append(FILE[line][65:])
            QRD_time_list.append(FILE[line][11:24])
            QRD_idx.append(line)

    for j in range(len(QRA_list)):
        QRA_data_t = np.fromstring(QRA_list[j], dtype=float, sep=',')
        QRA = np.vstack((QRA, QRA_data_t))
    QRA = QRA[1:,:]

    for k in range(len(QRD_list)):
        QRD_data_t = np.fromstring(QRD_list[k], dtype=float, sep=',')
        QRD = np.vstack((QRD, QRD_data_t))
    QRD = QRD[1:, :]

    for l in range(len(QRA_time_list)):
        h, m, s = QRA_time_list[l].split(':')
        QRA_time.append(int(h) * 3600 + int(m) * 60 + float(s))

    for m in range(len(QRD_time_list)):
        h, m, s = QRD_time_list[m].split(':')
        QRD_time.append(int(h) * 3600 + int(m) * 60 + float(s))

    return QRA, QRA_time, QRA_idx, QRD, QRD_time, QRD_idx


# Read log file check FT reading
def readLog_FT(log_file_path, startLine, endLine):
    log_file = open(log_file_path, 'r')
    FILE = log_file.readlines()
    FT_time_list = []
    FT_list = []
    FTJ_time_list = []
    FTJ_list = []
    FT = []
    FTJ = []
    FT_time = []
    FTJ_time = []
    for line in range(startLine,endLine):
        if "FT:" in FILE[line]:
            FT_list.append(FILE[line][49:])
            FT_time_list.append(FILE[line][11:24])
        if "FTJ:" in FILE[line]:
            FTJ_list.append(FILE[line][50:])
            FTJ_time_list.append(FILE[line][11:24])
    #print('FT list is', FT_list)

    for j in range(len(FT_list)):
        FT_data_t = np.fromstring(FT_list[j], dtype=float, sep=',')
        #print('FT_data_t is', FT_data_t)
        FT.append(FT_data_t)
        #FT = np.vstack((FT, FT_data_t))
    FT = np.asarray(FT)

    for k in range(len(FTJ_list)):
        FTJ_data_t = np.fromstring(FTJ_list[k], dtype=float, sep=',')
        FTJ.append(FTJ_data_t)
        #FTJ = np.vstack((FTJ, FTJ_data_t))
    FTJ = np.asarray(FTJ)

    for l in range(len(FT_time_list)):
        h, m, s = FT_time_list[l].split(':')
        FT_time.append(int(h) * 3600 + int(m) * 60 + float(s))

    for m in range(len(FTJ_time_list)):
        h, m, s = FTJ_time_list[m].split(':')
        FTJ_time.append(int(h) * 3600 + int(m) * 60 + float(s))

    return FT, FT_time, FTJ, FTJ_time


# Read log file check tracker_kdl
def readLog_tracker(log_file_path):
    log_file = open(log_file_path, 'r')
    FILE = log_file.readlines()
    tracker_kdl = []
    for line in range(len(FILE)):
        find_tracker = 0
        if "tracker_kdl:" in FILE[line]:
            str = FILE[line][80:-1]
            tracker_kdl = np.fromstring(str, dtype=float, sep=',')
            find_tracker = 1
            break
    if find_tracker == 1:
        print('find tracker kdl in file')
    else:
        print('Error, tracker_kdl is not found, please check file')
    return tracker_kdl


# Read log file check schunk_kdl
def readLog_ga(log_file_path):
    log_file = open(log_file_path, 'r')
    FILE = log_file.readlines()
    kdl = []
    for line in range(len(FILE)):
        find_tracker = 0
        if "schunk_kdl:" in FILE[line]:
            str = FILE[line][79:-1]
            kdl = np.fromstring(str, dtype=float, sep=',')
            find_tracker = 1
            break
    if find_tracker == 1:
        print('find schunk kdl in file')
    else:
        print('Error, schunk_kdl is not found, please check file')
    return kdl


# Read log file check dfc
def readLog_dfc_upper(log_file_path):
    log_file = open(log_file_path, 'r')
    FILE = log_file.readlines()
    kdl = []
    for line in range(len(FILE)):
        find_tracker = 0
        if "T_sf_df_nominal_upper:" in FILE[line]:
            str = FILE[line][95:-1]
            kdl = np.fromstring(str, dtype=float, sep=',')
            find_tracker = 1
            break
    if find_tracker == 1:
        print('find dfc_nominal_upper in file')
    else:
        print('Error, dfc_nominal_upper is not found, please check file')
    return kdl


# Read log file check dfc at drill process
def readLog_dfc_upper_in_drill(log_file_path):
    log_file = open(log_file_path, 'r')
    FILE = log_file.readlines()
    kdl = []
    for line in range(len(FILE)):
        find_tracker = 0
        if "Set T_sf_df to" in FILE[line]:
            str = FILE[line][102:-1]
            kdl = np.fromstring(str, dtype=float, sep=',')
            find_tracker = 1
            break
    if find_tracker == 1:
        print('find dfc_nominal_upper in file')
    else:
        print('Error, dfc_nominal_upper is not found, please check file')
    return kdl


# Read log file check drill length
def readLog_drill_length(log_file_path):
    log_file = open(log_file_path, 'r')
    FILE = log_file.readlines()
    kdl = []
    for line in range(len(FILE)):
        find_tracker = 0
        if "Drill bit length" in FILE[line]:
            str = FILE[line][70:-1]
            tem = np.fromstring(str, dtype=float, sep=',')
            find_tracker = 1
            kdl.append(tem)
    if find_tracker == 1:
        print('find drill bt length in file')
    else:
        print('Error, drill bit length is not found, please check file')
    return kdl


# check the idx of the line which contains 'str'
# The output is an array with all idx.
def check_idx_log(log_file_path, str):
    # str = "Set Jog To mode"
    log_file = open(log_file_path, 'r')
    FILE = log_file.readlines()
    idx = []
    for line in range(len(FILE)):
        if str in FILE[line]:
            idx_tem = line
            idx.append(idx_tem)
    #idx_end = len(FILE)-1
    #idx.append(idx_end)
    idx = np.asarray(idx)
    return idx


# check the idx of the line which contains 'str'
# The output is an array with all idx plus the last line of the file
def check_idx_log_endLine(log_file_path, str):
    # str = "Set Jog To mode"
    log_file = open(log_file_path, 'r')
    FILE = log_file.readlines()
    idx_end = len(FILE) -1
    idx = []
    for line in range(len(FILE)):
        if str in FILE[line]:
            idx_tem = line
            idx.append(idx_tem)
    idx.append(idx_end)
    idx = np.asarray(idx)
    return idx


# check the idx of the line which contains 'str'
# The output is an array with all idx
# idx[2] is the paused status at pose 2
def check_idx_log_api_mode_pause(log_file_path, str="API_CM_MODE switched to 576"):
    log_file = open(log_file_path, 'r')
    FILE = log_file.readlines()
    idx_end = len(FILE) -1
    idx = []
    for line in range(len(FILE)):
        if str in FILE[line]:
            idx_tem = line
            idx.append(idx_tem)
    #idx.append(idx_end)
    idx = np.asarray(idx)
    return idx


# check the idx of the line which contains 'str'
# The output is an array with all idx
# idx[1] is the start jogging to pose 2.
def check_idx_log_api_mode_start(log_file_path, str="API_CM_MODE switched to 592"):
    # str = "Set Jog To mode"
    log_file = open(log_file_path, 'r')
    FILE = log_file.readlines()
    idx_end = len(FILE) -1
    idx = []
    for line in range(len(FILE)):
        if str in FILE[line]:
            idx_tem = line
            idx.append(idx_tem)
    #idx.append(idx_end)
    idx = np.asarray(idx)
    return idx


# read utilTracker.log
def read_utiltracker_log(file):
    log_file = open(file, 'r')
    FILE = log_file.readlines()
    QT = []
    QT_data = []
    for line in range(74, len(FILE)):
        if "Q :" in FILE[line]:
            QT.append(FILE[line][63:])

    for j in range(len(QT)):
        QT_data_t = np.fromstring(QT[j], dtype=float, sep=' ')
        QT_data.append(QT_data_t)

    return np.asarray(QT_data)


# read barset_measurements
def read_barset(file):
    log_file = open(file, 'r')
    FILE = log_file.readlines()
    for line in range(len(FILE)):
        if "Bar Short" in FILE[line]:
            sb_s = FILE[line][10:-1]
            sb = np.fromstring(sb_s, dtype=float, sep=' ')
        if "Bar Medium" in FILE[line]:
            mb_s = FILE[line][11:-1]
            mb = np.fromstring(mb_s, dtype=float, sep=' ')
        if "Bar Long" in FILE[line]:
            lb_s = FILE[line][9:-1]
            lb = np.fromstring(lb_s, dtype=float, sep=' ')
    barlength=np.zeros(3)
    barlength[0] = sb
    barlength[1] = mb
    barlength[2] = lb
    return barlength