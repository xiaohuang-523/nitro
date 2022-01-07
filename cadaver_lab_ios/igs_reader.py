import Readers as Yomiread
import Writers as Yomiwrite
import fileDialog

folder = fileDialog.select_folder()
file = folder + '/points.igs'
data = Yomiread.read_csv_D_type(file, 4, 20)
print('data is', data)
Yomiwrite.write_csv_matrix(folder + '/points.txt', data)