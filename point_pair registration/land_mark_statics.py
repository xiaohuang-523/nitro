import numpy as np
import Readers as Yomiread
import statistics_analysis as sa

path = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\Edentulous registration error analysis\\Case report\\"
file = "Landmark Value Summary.csv"

data = Yomiread.read_csv(path + file, 3, -1)
land_mark = []
for i in range(np.shape(data)[1]):
    for element in data[:,i]:
        land_mark.append(element)

sa.fit_models(land_mark[:-3])

print('shape of land_mark is', np.shape(land_mark))