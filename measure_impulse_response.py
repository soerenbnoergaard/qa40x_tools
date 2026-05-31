import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import qa40x as qa
import algorithm

def measure(label="", amplitude_dBV=-20, num_points=256 * 1024):
    inst = qa.Qa40x()
    assert amplitude_dBV < 10
    fs = qa.SAMPLE_RATE_Hz

    # Using a ESS waveform instead of the built-in ExpoChirp, to be in better control.
    # Still using a 10-octave sweep up to fs/2
    #
    # farina2007advancements_in_impulse_response_measurements_by_sine_sweeps

    f2 = fs/2
    f1 = f2 / (2**10)

    # Leaving some silence in the beginning and the end
    x = np.zeros(num_points)
    n_start = 1 * num_points // 8
    n_stop = 7 * num_points // 8
    N = n_stop - n_start
    A = 10**(amplitude_dBV / 20)
    # R = ramp(10000, 0, N)
    x[n_start:n_stop] = A * algorithm.expochirp(f1, f2, N, fs)
    t = np.arange(num_points) / fs

    inst.set_buffer_size(N)
    inst.acquire_custom_waveform(x)
    meas = inst.measure_recorded_waveform()
    data = {
        "frequency_start_Hz": f1,
        "frequency_stop_Hz": f2,
        "time_s": t.tolist(),
        "input_V": x.tolist(),
        "output_left_V": meas["Left"],
        "output_right_V": meas["Right"],
        "index_start": n_start,
        "index_stop": n_stop,
    }
    qa.dump_data(qa.generate_filename(label, "impulse_response"), data)
    return data

def ramp(ramp_up, ramp_down, num_samples):
    r = np.ones(num_samples)

    for n in range(ramp_up):
        r[n] = n * 1 / ramp_up

    for n in range(ramp_down):
        r[num_samples - n - 1] = n * 1 / ramp_down

    return r

def analyze(data):
    t = data["time_s"]
    x = data["input_V"]
    y = data["output_left_V"]
    n_start = data["index_start"]
    n_stop = data["index_stop"]
    f_start = data["frequency_start_Hz"]
    f_stop = data["frequency_stop_Hz"]
    N = len(x)

    fig, axs = plt.subplots(4, 1, figsize=(8, 8))
    ax = axs[0]
    ax.plot(t, x)
    ax.plot(t, y)

    # Extract chirp only
    x = x[n_start:n_stop]
    y = y[n_start:n_stop]

    h_est = algorithm.deconvolve(x, y, f_start, f_stop)

    ax = axs[2]
    ax.plot(h_est)

    # Calculate frequency response
    H_est = np.abs(np.fft.fft(h_est))
    H_est = H_est[0:len(H_est)//2]
    f_est = np.linspace(0, 24000, len(H_est))

    ax = axs[3]
    ax.semilogx(f_est, 20*np.log10(H_est))
    ax.set_xlim(20, 20e3)
    # ax.set_ylim(-170, 20)

    fig.tight_layout()

    # # TODO: Handle pre-ringing


if __name__ == "__main__":
    # data = measure()
    # analyze(data)

    analyze(qa.load_data("_impulse_response_2026-05-25_141319.json.gz"))
    plt.show()
