import numpy as np
import matplotlib.pyplot as plt

import algorithm
import soundfile as sf

def test_deconvolve_reaverb_compare():
    x, fs_x = sf.read("test_data/reaverb_sweep_source.wav")
    y, fs_y = sf.read("test_data/reaverb_sweep_recording.wav")
    h, fs_h = sf.read("test_data/reaverb_sweep_deconvolved.wav")

    # TODO: What are the start and stop frequencies in reaverb?
    f_start = 0
    f_stop = 0

    # TODO: Compare with algorithm.deconvolve()

    fig, ax = plt.subplots()
    ax.plot(h)
    plt.show()