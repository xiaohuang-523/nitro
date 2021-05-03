import Readers as Yomiread

file = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\IOS_scan_raw_data\\IOS Splint\\v3\\Faro Measurement\\IOS_splint_v3_trial3_molar1.csv"

x = Yomiread.read_faro_ios_splint_measurement(file)
print(x)