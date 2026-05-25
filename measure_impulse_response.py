import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import qa40x as qa

def expochirp(f1, f2, N, fs):
    T = N / fs
    t = np.arange(N) / fs
    w1 = 2 * np.pi * f1
    w2 = 2 * np.pi * f2
    return np.sin(
        (w1 * T) / (np.log(w2 / w1)) *
        (np.exp(t / T * np.log(w2 / w1)) - 1)
    )

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
    x[n_start:n_stop] = A * expochirp(f1, f2, N, fs)
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

    # Construct the "inverse filter" from the input and apply amplitude modulation to invert the pink noise slope
    # Amplitude modulation: Starts at 0 dB, ends at -6 * log2(w2 / w1).
    #
    # - farina2000simultaneous_measurement_of_impulse_response_and_distortion_with_a_swept-sine_technique
    # - holters2009impulse_response_measurement_techniques_and_their_applicability_in_the_real_world

    log2 = lambda _x: np.log(_x) / np.log(2)
    L = n_stop - n_start
    a_start_dB = 0
    a_stop_dB = a_start_dB - 6 * log2(f_stop / f_start)
    a_mod = 10**(np.linspace(a_start_dB, a_stop_dB, L) / 20)

    x_inv_section = a_mod * np.flipud(x[n_start:n_stop])
    x_inv = np.zeros(N)
    x_inv[N - n_stop - 1:N - n_start - 1] = x_inv_section

    ax = axs[1]
    ax.plot(x)
    ax.plot(x_inv)

    # Do deconvolution to obtain impulse response
    h_est = scipy.signal.convolve(y, x_inv, "full")
    n_est_mid = len(h_est) // 2
    n_est = (np.arange(len(h_est)) - n_est_mid)
    n_est_low = n_est_mid - 1000
    n_est_high = n_est_mid + 10000

    ax = axs[2]
    ax.plot(n_est, h_est)
    ax.axvline(n_est_low - n_est_mid)
    ax.axvline(n_est_high - n_est_mid)
    ax.set_xlim(n_est_low - n_est_mid - 100, n_est_high - n_est_mid + 100)

    # Extract
    h_est = h_est[n_est_low : n_est_high]

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
    data = measure()
    analyze(data)

    # analyze(qa.load_data("_impulse_response_2026-05-25_140658.json.gz"))
    plt.show()
