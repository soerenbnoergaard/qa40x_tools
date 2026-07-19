import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import soundfile as sf
import qa40x as qa
import algorithm
from matplotlib.ticker import EngFormatter

plt.style.use("bmh")

def measure(label="", amplitude_dBV=-20, num_points=256 * 1024, sample_rate_Hz=96000):
    inst = qa.Qa40x()
    assert amplitude_dBV < 10
    fs = sample_rate_Hz
    inst.set_sample_rate_Hz(fs)

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
        "label": label,
        "sample_rate_Hz": fs,
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
    label = data["label"]
    t = data["time_s"]
    x = data["input_V"]
    y = data["output_left_V"]
    n_start = data["index_start"]
    n_stop = data["index_stop"]
    f_start = data["frequency_start_Hz"]
    f_stop = data["frequency_stop_Hz"]
    fs = data["sample_rate_Hz"]
    N = len(x)
    
    fig, axs = plt.subplots(4, 1, figsize=(8, 8))
    ax = axs[0]
    ax.plot(t, x, color="C0")
    ax.set(xlabel="Time [s]", ylabel="Input [V]")

    ax = axs[1]
    ax.plot(t, y, color="C1")
    ax.set(xlabel="Time [s]", ylabel="Output [V]")

    h_est = algorithm.deconvolve(x, y, n_start, n_stop, f_start, f_stop)
    nr = np.arange(len(h_est)) - len(h_est) // 2

    ax = axs[2]
    ax.plot(nr, h_est, color="C2")
    ax.set(xlabel="Time [samples]", ylabel="Impulse\nresponse [V]")

    # Calculate frequency response from the last half of h_est where the "fundamental" IR is located.
    _h = h_est.copy()
    _mid = len(_h) // 2
    _stop = _mid + (n_stop - n_start)
    _h = _h[_mid:_stop]
    _N = len(_h)
    ax.axvspan(nr[_mid], nr[_stop], color="C3", alpha=0.3)
    
    H_est = np.abs(np.fft.rfft(_h, norm="ortho"))
    f_est = np.linspace(0, fs/2, len(H_est))

    ax = axs[3]
    ax.semilogx(f_est, 20*np.log10(H_est), color="C3")
    ax.set_xlim(20, 20e3)
    ax.set(xlabel="Frequency [Hz]", ylabel="Frequency\nresponse [dB]")
    ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    ax.xaxis.set_major_formatter(EngFormatter(sep=""))
    # ax.set_ylim(-170, 20)

    # Reference exported from QA40x
    # f_ref, H_ref = np.loadtxt("boss_katana_reference_freq_resp.csv", skiprows=6, delimiter=",", usecols=(0, 1)).T
    # H_ref -= 10
    # ax.semilogx(f_ref, H_ref, color="C4")
    
    # TODO: Handle pre-ringing?

    fig.suptitle(label)
    fig.tight_layout()

    # Save results (export the full IR as float and manually crop/normalize it afterwards)
    basename = label
    print(f"Saving results for {basename}...")
    fig.savefig(basename + ".png")
    sf.write(basename + "_raw.wav", h_est, fs, format="WAV", subtype="FLOAT")

if __name__ == "__main__":
    # data = measure(
    #     label="boss_katana_celstion_a-type_sm57",
    #     amplitude_dBV=-40,
    #     num_points=int(2**18),
    #     sample_rate_Hz=96000,
    # )
    # analyze(data)

    # analyze(qa.load_data("_impulse_response_2026-05-25_141319.json.gz"))
    analyze(qa.load_data("boss_katana_celstion_a-type_sm57_impulse_response_2026-07-19_115717.json.gz"))
    plt.show()
