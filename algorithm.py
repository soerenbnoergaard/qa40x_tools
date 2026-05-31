import numpy as np
import scipy.signal

def expochirp(f1, f2, N, fs):
    T = N / fs
    t = np.arange(N) / fs
    w1 = 2 * np.pi * f1
    w2 = 2 * np.pi * f2
    return np.sin(
        (w1 * T) / (np.log(w2 / w1)) *
        (np.exp(t / T * np.log(w2 / w1)) - 1)
    )

def log2(x):
    return np.log(x) / np.log(2)

def inverse_filter(x, f_start, f_stop):
    # Construct the "inverse filter" from the input and apply amplitude modulation to invert the pink noise slope
    # Amplitude modulation: Starts at 0 dB, ends at -6 * log2(w2 / w1).
    #
    # - farina2000simultaneous_measurement_of_impulse_response_and_distortion_with_a_swept-sine_technique
    # - holters2009impulse_response_measurement_techniques_and_their_applicability_in_the_real_world

    # Assuming x contains the full sweep from f_start to f_stop
    L = len(x)
    a_start_dB = 0
    a_stop_dB = a_start_dB - 6 * log2(f_stop / f_start)
    a_mod = 10**(np.linspace(a_start_dB, a_stop_dB, L) / 20)
    x_inv = a_mod * np.flipud(x)
    return x_inv

def deconvolve(x, y, f_start, f_stop):
    """x=chirp source, y=chirp response"""

    x_inv = inverse_filter(x, f_start, f_stop)
    h_est = scipy.signal.convolve(y, x_inv, "full")

    return h_est