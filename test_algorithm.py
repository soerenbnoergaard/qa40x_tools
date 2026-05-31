import pytest

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

import algorithm
import soundfile as sf

@pytest.mark.skip(reason="Reaverb sweep is not exponential")
def test_reproduce_reaverb_chirp():
    x1, fs = sf.read("test_data/reaverb_sweep_source.wav")

    # It looks like the Reaverb sweep is not actually exponential, but has a bit of a curve
    # So direct comparison is not possible
    # But it looks like a 10-octave sweep ending at fs/2 is fairly close
    f_stop = fs / 2
    f_start = f_stop / (2**10)
    x2 = algorithm.expochirp(f_start, f_stop, len(x1), fs)

    # fig, ax = plt.subplots()
    # ax.specgram(x1, Fs=fs, alpha=0.5, cmap="Blues")
    # ax.specgram(x2, Fs=fs, alpha=0.5, cmap="Oranges")
    # ax.set_yscale("log")
    # ax.set_ylim(20, 24000)
    # plt.show()

    for (_x1, _x2) in zip(x1, x2):
        assert _x1 == pytest.approx(_x2, 0.001)

def test_decolvolve_fir_filter():
    fs = 48000
    Nc = 512
    h = scipy.signal.firwin(Nc, cutoff=1000, fs=fs)

    # Measure system with a chirp
    N = 65536
    f_stop = fs/2
    f_start = f_stop / (2**15)
    x = algorithm.expochirp(f_start, f_stop, N, fs)
    y = scipy.signal.convolve(x, h, "full")

    # Deconvolve to estimate h, extract filter taps, and normlize
    h_est = algorithm.deconvolve(x, y, f_start, f_stop)
    n_mid = len(h_est) // 2
    h_est = h_est[n_mid - Nc//2 : n_mid + Nc//2]
    h_est /= np.sum(h_est)

    np.testing.assert_allclose(h, h_est, atol=0.001)

    # fig, ax = plt.subplots()
    # ax.plot(h, linewidth=5, color="blue")
    # ax.plot(h_est, linewidth=2, color="yellow")
    # plt.show()
