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
    # ax.plot(h_est, linewidth=2, color="orange")
    # plt.show()

def test_decolvolve_distorted_fir_filter():
    fs = 48000
    Nc = 512
    h = scipy.signal.firwin(Nc, cutoff=1000, fs=fs)

    def distort(x):
        return np.tanh(1.5 * x + 0.2) / 1.5 - 0.2
        # return np.tanh(1.5 * x + 0.5) / 1.5 - 0.5

    # Measure system with a chirp
    N = 65536
    f_stop = fs/2
    f_start = f_stop / (2**15)
    x = algorithm.expochirp(f_start, f_stop, N, fs)
    x_dist = distort(x)
    y = scipy.signal.convolve(x_dist, h, "full")

    # Apply deconvolution
    h_est = algorithm.deconvolve(x, y, f_start, f_stop)

    # Extract fundamental and harmonics
    # chan2010swept_sine_chirps_for_measuring_impulse_response
    def extract(h, harmonic=1, num_samples=Nc, sweep_length=N, f_start=f_start, f_stop=f_stop):
        n0 = len(h) // 2
        T = sweep_length
        delta_tau_gn = - T * np.log(harmonic) / np.log(f_stop / f_start)
        n_mid = int(np.round(n0 + delta_tau_gn))
        return h[n_mid - num_samples//2 : n_mid + num_samples//2].copy()

    h1 = extract(h_est, 1)
    h2 = extract(h_est, 2)
    h3 = extract(h_est, 3)
    h4 = extract(h_est, 4)
    h5 = extract(h_est, 5)

    # Normalize
    h_norm = np.sum(h1)
    h1 /= h_norm
    h2 /= h_norm
    h3 /= h_norm
    h4 /= h_norm
    h5 /= h_norm

    # Check that the fundamental response is still as expected
    np.testing.assert_allclose(h, h1, atol=0.001)

    # # Plot impulse resopnses
    # fig, ax = plt.subplots()
    # ax.plot(h, linewidth=5, label="target")
    # ax.plot(h1, linewidth=2, label="h1")
    # ax.plot(h2, linewidth=2, label="h2")
    # ax.plot(h3, linewidth=2, label="h3")
    # ax.plot(h4, linewidth=2, label="h4")
    # ax.plot(h5, linewidth=2, label="h5")
    # ax.legend()

    # # Plot frequency responses
    # fig, ax = plt.subplots()
    # _f = np.linspace(0, fs, Nc)
    # ax.semilogx(_f, 20*np.log10(np.abs(np.fft.fft(h))), linewidth=5, label="target")
    # ax.semilogx(_f, 20*np.log10(np.abs(np.fft.fft(h1))), linewidth=2, label="h1")
    # ax.semilogx(_f, 20*np.log10(np.abs(np.fft.fft(h2))), linewidth=2, label="h2")
    # ax.semilogx(_f, 20*np.log10(np.abs(np.fft.fft(h3))), linewidth=2, label="h3")
    # ax.semilogx(_f, 20*np.log10(np.abs(np.fft.fft(h4))), linewidth=2, label="h4")
    # ax.semilogx(_f, 20*np.log10(np.abs(np.fft.fft(h5))), linewidth=2, label="h5")
    # ax.set_xlim(10, fs/2)
    # ax.legend()

    # plt.show()