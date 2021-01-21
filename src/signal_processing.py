from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

# High pass filter
# Modified based on
# https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
# https://en.wikipedia.org/wiki/Normalized_frequency_(unit)

# Guide 1. The use of signal.butter() function
#   Method 1. Normalize the cutoff frequency.
#       nyq = 0.5 * fs
#       normal_cutoff = cutoff/nyq
#       signal.butter(order, normal_cutoff, btype, analog, output)
#   Method 2. Specify sampling rate fs.
#       signal.butter(order, cutoff, btype, analog, output, fs=fs)
#
#
# Guide 2. Apply filter to data
#   Method 1. Use signal.sosfilt() function
#       using cascaded second-order sections
#       Solve the sos solution from signal.butter()
#       sos = signal.butter(order, cutoff, ... , output='sos')
#       filtered = signal.sosfilt(sos, data)
#
#   Method 2. Use signal.lfilter() function
#       using IIR or FIR filter. b is the numerator and a is the denominator of the IIR filter.
#       solve the b,a solution from signal.butter()
#       b, a = signal.butter(order, cutoff, ... , output='ba')
#       filtered = signal.lfiter(b, a, data)
#
# Guide 3. Frequency response
#   signal.freqs(b,a)       compute the frequency response of analog filter
#   signal.freqz(b,a)       compute the frequency response of digital filter
#   The returned results are w and h.
#      w: analog: angular frequencies
#         digital: normalized frequencies, radians/sample
#      h: frequency response.
#
# Guide 4. Angular frequency
#   w = 2*pi*f
#       w: angular frequency (angular velocity) radians/s
#       f: ordinary frequency hertz
#   1 radian/s = 1/2*pi Hz
#
# Guide 5. Normalized frequency
#   In digital signal processing (DSP)
#   f (cycles/sec) = f/fs, where fs = 1/T (samples/sec)
#
#   if fs is real number, Nyquist frequency is fs/2 is the maximum frequency that can be unambiguously represented by
#   digital data.
#   if fs is complex number, the maximum frequency is fs
#
#   In scipy.signal.butter, the frequency is normalized by fs/2
#
# Guide 6. Plot frequency response
#   plt.semilogx()   Take log scale on x axis
#


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    ba = signal.butter(order, normal_cutoff, btype='high', analog=False, output='ba')
    return ba


def butter_highpass_filter(data, cutoff, fs, order=5):
    ba = butter_highpass(cutoff, fs, order=order)
    filtered = signal.lfilter(ba[0], ba[1], data)
    return filtered


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    ba = signal.butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    return ba


def butter_lowpass_filter(data, cutoff, fs, order=5):
    ba = butter_lowpass(cutoff, fs, order=order)
    filtered = signal.lfilter(ba[0], ba[1], data)
    return filtered


def plot_frequency_response(w,h,fs,cutoff):
    plt.semilogx(0.5 * fs * w / np.pi, 20 * np.log10(np.abs(h)))
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(cutoff, color='green')  # cutoff frequency


def highpass_fft(data, cutoff, fs):
    time_step = 1/fs
    x = np.copy(data)
    fft = fftpack.fft(x)
    power = np.abs(fft)
    x_freq = fftpack.fftfreq(data.size, d = time_step)
    plt.figure()
    plt.plot(x_freq, power)
    pos_mask = np.where(x_freq > 0)
    freqs = x_freq[pos_mask]
    peak_freq = freqs[power[pos_mask].argmax()]

    axes = plt.axes([0.55, 0.3, 0.3, 0.5])
    plt.title('Peak frequency')
    plt.plot(freqs[:8], power[:8])
    plt.setp(axes, yticks=[])

    high_freq_fft = fft.copy()
    high_freq_fft[np.abs(x_freq) > peak_freq] = 0
    filtered_sig = fftpack.ifft(high_freq_fft)

    plt.figure(figsize=(6, 5))
    plt.plot(range(len(x)), data, label='Original signal')
    plt.plot(range(len(x)), filtered_sig, linewidth=3, label='Filtered signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    plt.legend(loc='best')




