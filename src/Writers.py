import csv
import numpy as np

# write csv file with string
def write_csv_array2string(file_name, data):
    file = open(file_name, "w")
    for line in data:
        file.write(np.array2string(line) + "\n")
    file.close()


# write csv file with string
def write_csv_string(file_name, data):
    file = open(file_name, "w")
    file.write(data + "\n")
    file.close()


# write csv file with string
def write_csv_string_simple(file_name, data):
    file = open(file_name, "w")
    for line in data:
        file.write(line + "\n")
    file.close()

# write csv file with data
def write_csv_array(file_name: object, data: object, fmt: object = '%.4f') -> object:
    np.savetxt(file_name, data, delimiter=",", newline="\n", fmt = fmt)


# write csv file with data (matrix)
def write_csv_matrix(file_name, data, fmt = '%.4f', delim = ","):
    f = open(file_name, 'w')
    data = np.asarray(data)
    if len(np.shape(data)) > 1:
        for data_tem in np.asarray(data):
            np.savetxt(f, data_tem.reshape(1, data_tem.shape[0]), delimiter=delim, fmt = fmt)
    else:
        np.savetxt(f, data.reshape(1, len(data)), delimiter=delim, fmt = fmt)
        #f.write("\n")
    f.close()

# write csv file with data (matrix)
def write_csv_array_append(file_name, data, fmt = '%.4f'):
    f = open(file_name, 'a')
    data = np.asarray(data)
    if len(np.shape(data)) > 1:
        for data_tem in np.asarray(data):
            np.savetxt(f, data_tem.reshape(1, data_tem.shape[0]), delimiter=",", fmt = fmt)
    else:
        np.savetxt(f, data.reshape(1, len(data)), delimiter=',', fmt = fmt)
    f.close()
    
# write csv file with data (matrix)
def write_csv_matrix_append(file_name, data, fmt = '%.4f'):
    f = open(file_name, 'a')
    data = np.asarray(data)
    if len(np.shape(data)) > 1:
        for data_tem in np.asarray(data):
            np.savetxt(f, data_tem.reshape(1, data_tem.shape[0]), delimiter=",", fmt = fmt)
    else:
        np.savetxt(f, data.reshape(1, len(data)), delimiter=',', fmt = fmt)
        #f.write("\n")
    f.close()