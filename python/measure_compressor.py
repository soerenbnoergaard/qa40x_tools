import gzip
import time
import requests
import struct
import base64
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

SAMPLE_RATE_Hz = 48000

def is_power_of_two(n):
    if n <= 0:
        return False
    return (n & (n - 1)) == 0

def floats_to_base64(floats):
    b = struct.pack("<" + "d" * len(floats), *floats)
    return base64.b64encode(b).decode("ascii")

def base64_to_floats(b64):
    b = base64.b64decode(b64)
    count = len(b) // 8
    return list(struct.unpack("<" + "d" * count, b))

def dump_data(filename, data):
    json_bytes = json.dumps(data).encode("utf-8")
    with gzip.open(filename, "w") as fh:
        fh.write(json_bytes)

def load_data(filename):
    with gzip.open(filename, "r") as fh:
        json_str = fh.read().decode("utf-8")
    return json.loads(json_str)

def generate_filename(filename_base, measurement_name, extension="json.gz"):
    return filename_base + "_" + measurement_name + "_" + time.strftime("%Y-%m-%d_%H%M%S") + "." + extension

class QA403:
    def __init__(self):
        self.set_buffer_size(65536)
        self.set_sample_rate_Hz(SAMPLE_RATE_Hz)

    def post(self, s, data=None):
        return requests.post(f"http://localhost:9402/{s}", json=data).json()

    def get(self, s):
        return requests.get(f"http://localhost:9402/{s}").json()

    def put(self, s):
        return requests.put(f"http://localhost:9402/{s}").json()
    
    def acquire(self):
        return self.post("Acquisition")

    def setup_gen1(self, enable, frequency_Hz, amplitude_dBV):
        return self.put("Settings/AudioGen/Gen1/{}/{}/{}".format(
            "On" if enable else "Off",
            frequency_Hz,
            amplitude_dBV,
        ))
        
    def measure_rms_dBV(self):
        return self.get("RmsDbv/20/20000")["Left"]

    def set_buffer_size(self, buffer_size):
        return self.put(f"Settings/BufferSize/{buffer_size}")

    def set_sample_rate_Hz(self, sample_rate_Hz):
        return self.put(f"Settings/SampleRate/{sample_rate_Hz}")

    def acquire_custom_waveform(self, waveform_V):
        N = len(waveform_V)
        assert is_power_of_two(N)
        self.set_buffer_size(N)
        s = floats_to_base64(waveform_V)
        data = {
            "Left": s,
            "Right": s,
        }
        self.post("Acquisition", data=data)

    def measure_recorded_waveform(self):
        waveform_base64 = self.get("/Data/Time/Input")["Left"]
        return base64_to_floats(waveform_base64)

def lerp(x, x0, y0, x1, y1):
    return (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)

def measure_compressor_curve(label, amplitude_min_dBV, amplitude_max_dBV, amplitude_step_dB, frequency_Hz=1000):
    assert (
        (amplitude_step_dB > 0 and amplitude_min_dBV < amplitude_max_dBV) or
        (amplitude_step_dB < 0 and amplitude_min_dBV > amplitude_max_dBV)
    )
    qa = QA403()
    inputs_dBV = []
    outputs_dBV = []

    input_dBV = amplitude_min_dBV
    while input_dBV <= amplitude_max_dBV:
        qa.setup_gen1(True, frequency_Hz, input_dBV)
        qa.acquire()
        output_dBV = float(qa.measure_rms_dBV())
        inputs_dBV.append(input_dBV)
        outputs_dBV.append(output_dBV)
        print(f"input={input_dBV:.1f} dBV, output={output_dBV:.1f} dBV, gain={output_dBV - input_dBV:.1f} dB")
        input_dBV += amplitude_step_dB

    # Save to JSON file
    data = {
        "label": label,
        "frequency_Hz": frequency_Hz,
        "input_dBV": [_x for _x in inputs_dBV],
        "output_dBV": [_x for _x in outputs_dBV],
    }
    dump_data(generate_filename(label, "compressor_curve"), data)
    return data

def analyze_compressor_curve(super_label, datasets):
    fig, ax = plt.subplots()
    for idx, data in enumerate(datasets):
        label = data["label"]
        Vi = np.array(data["input_dBV"])
        Vi = np.array(data["input_dBV"])
        Vo = np.array(data["output_dBV"])
        G = Vo - Vi

        # Find threshold as the 1 dB compression point
        G_comp = G[0] - 1
        Vi_comp = np.nan
        for n in range(1, len(G)):
            if G[n] < G_comp:
                Vi_comp = lerp(G_comp, G[n-1], Vi[n-1], G[n], Vi[n])
                break

        print(f"Threshold: Vi={Vi_comp:.1f} dBV, Vo={Vi_comp+G_comp:.1f} dBV, G={G_comp:.1f} dB")

        color = f"C{idx}"
        ax.plot(Vi, G, ".-", color=color, label=label + f"\n(threshold={Vi_comp:.1f} dBV)")
        ax.plot(Vi_comp, G_comp, "o", color=color)

    ax.set_xlabel("Input [dBV]")
    ax.set_ylabel("Gain [dB]")
    ax.legend(loc="lower left", ncols=1)
    fig.tight_layout()
    fig.savefig(generate_filename(super_label, "compressor_curve", "png"))

def measure_compressor_attack_release(is_release, label, voltage_start_dBV=-40, voltage_stop_dBV=-20, frequency_Hz=1000):
    measurement_label = "compressor_release" if is_release else "compressor_attack"
    qa = QA403()

    if is_release:
        buffer_size = 262144
    else:
        buffer_size = 65536

    qa.set_buffer_size(buffer_size)

    V_start = 10**(voltage_start_dBV/20.0)
    V_stop = 10**(voltage_stop_dBV/20.0)

    def generate_waveform():
        t = np.arange(buffer_size) / SAMPLE_RATE_Hz
        waveform = np.exp(2j * np.pi * frequency_Hz * t)

        n_transient = 1*buffer_size//8
        waveform[0:n_transient] *= V_start
        waveform[n_transient:] *= V_stop
        return waveform

    Vi = generate_waveform()

    qa.acquire_custom_waveform(np.real(Vi))
    Vo_i = np.array(qa.measure_recorded_waveform())
    time.sleep(3)
    qa.acquire_custom_waveform(np.imag(Vi))
    Vo_q = np.array(qa.measure_recorded_waveform())
    Vo = Vo_i + 1j*Vo_q

    data = {
        "label": label,
        "frequency_Hz": frequency_Hz,
        "input_real_V": np.real(Vi).tolist(),
        "input_imag_V": np.imag(Vi).tolist(),
        "output_real_V": np.real(Vo).tolist(),
        "output_imag_V": np.imag(Vo).tolist(),
    }
    dump_data(generate_filename(label, measurement_label), data)
    return data

def measure_compressor_attack(*args, **kwargs):
    return measure_compressor_attack_release(False, *args, **kwargs)
    
def measure_compressor_release(*args, **kwargs):
    return measure_compressor_attack_release(True, *args, **kwargs)

def analyze_compressor_attack_release(is_release, super_label, datasets, use_hilbert=False):
    measurement_label = "compressor_release" if is_release else "compressor_attack"
    fig, ax = plt.subplots()

    ymin = None
    ymax = None

    for idx, data in enumerate(datasets):
        label = data["label"]
        f = data["frequency_Hz"]

        if use_hilbert:
            from scipy.signal import hilbert
            Vi = hilbert(np.array(data["input_real_V"]) )
            Vo = hilbert(np.array(data["output_real_V"]))
        else:
            Vi = np.array(data["input_real_V"]) + 1j*np.array(data["input_imag_V"])    
            Vo = np.array(data["output_real_V"]) + 1j*np.array(data["output_imag_V"])    

        buffer_size = len(Vo)

        # Extract amplitude envelopes
        Ai = np.abs(Vi)
        Ao = np.abs(Vo)

        # Measurement points
        n_transient = 1*buffer_size//8
        n_begin = 1*buffer_size//16
        n_end = 15*buffer_size//16

        if is_release:
            n_low = n_end
            n_high = n_begin
        else:
            n_low = n_begin
            n_high = n_end

        # Measure small signal gain
        G_begin = Ao[n_begin] / Ai[n_begin]
        print(f"Initial signal gain = {20*np.log10(G_begin)} dB")
        Ao_end_exp = G_begin * Ai[n_end]
        print(f"Expected settling voltage = {20*np.log10(Ao_end_exp):.1f} dBV")
        Ao_end = Ao[n_end]
        print(f"Measured settling voltage = {20*np.log10(Ao_end):.1f} dBV")
        print(f"Gain reduction = {20*np.log10(Ao_end_exp/Ao_end):.1f} dB")

        # Measure attack time
        GR = Ao_end_exp - Ao_end

        # Method 1: Time from transient until 90% of the gain reduction is effective
        # Ao_target = Ao_end + (0.1 * GR)

        # Method 2: Measure attack time as: Time from transient until 63.2% of the gain reduction is reached (RC time constant method)
        Ao_target = Ao_end_exp - 0.632*GR
        print(f"Search target = {20*np.log10(Ao_target):.1f} dB")

        def compare(value, target):
            if is_release:
                return (value < target)
            else:
                return (value > target)

        n_target = n_transient
        for n in reversed(range(n_transient, n_end)):
            if compare(Ao[n], Ao_target):
                n_target = n
                break
        n_dur = n_target - n_transient
        t_dur = n_dur / SAMPLE_RATE_Hz
        if is_release:
            search_label = f"release={t_dur * 1e3:.3f} ms"
        else:
            search_label = f"attack={t_dur * 1e3:.3f} ms"
        print(f"Measured {search_label}")

        color = f"C{idx}"
        f = lambda x: 20*np.log10(np.abs(x))

        t = (np.arange(len(Ao)) - n_transient) / SAMPLE_RATE_Hz * 1000
        ax.plot(t, f(Ao), color=color, label=f"{label}\n{search_label}")
        ax.plot(t[n_transient], f(Ao_end_exp), "s", color=color)
        ax.plot(t[n_end], f(Ao_end), "s", color=color)
        ax.plot([t[n_target]], [f(Ao_target)], "o", color=color)

        # Autoscale
        if is_release:
            ymin_now = f(Ao_end_exp)
            ymax_now = f(Ao_end)
        else:
            ymin_now = f(Ao_end)
            ymax_now = f(Ao_end_exp)

        ymin = ymin_now if ymin is None or ymin_now < ymin else ymin
        ymax = ymax_now if ymax is None or ymax_now > ymax else ymax

    ax.legend(loc="lower right" if is_release else "upper right", ncols=1)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Output amplitude [dBV]")
    ax.set_ylim(ymin - 3, ymax + 3)
    fig.tight_layout()
    fig.savefig(generate_filename(super_label, measurement_label, "png"))


def analyze_compressor_attack(*args, **kwargs):
    return analyze_compressor_attack_release(False, *args, **kwargs)
    
def analyze_compressor_release(*args, **kwargs):
    return analyze_compressor_attack_release(True, *args, **kwargs)


LABEL = "RNC1773_T-20_R25_A0002_R5000_G0_SNon"
THRESHOLD_dBV = -20

# CURVE
# Recommended settings: attack=fast, release=fast

#data = measure_compressor_curve(LABEL, THRESHOLD_dBV - 30, THRESHOLD_dBV + 20, 1)
#analyze_compressor_curve(LABEL, [data])

#datasets = []
#datasets.append(load_data("RNC1773_T-20_R01_A01_R005_G0_SNoff_compressor_curve_2026-05-15_122929.json.gz"))
#datasets.append(load_data("RNC1773_T-20_R02_A01_R005_G0_SNoff_compressor_curve_2026-05-15_123204.json.gz"))
#datasets.append(load_data("RNC1773_T-20_R06_A01_R005_G0_SNoff_compressor_curve_2026-05-15_123527.json.gz"))
#datasets.append(load_data("RNC1773_T-20_R10_A01_R005_G0_SNoff_compressor_curve_2026-05-15_123716.json.gz"))
#datasets.append(load_data("RNC1773_T-20_R25_A01_R005_G0_SNoff_compressor_curve_2026-05-15_124454.json.gz"))
#analyze_compressor_curve("RNC1773", datasets)


# ATTACK
# Recommended settings: ratio=high, release=fast, Vstart=Vthres-20, Vstop=Vthres+20

#data = measure_compressor_attack(LABEL, THRESHOLD_dBV - 20, THRESHOLD_dBV + 20)
#analyze_compressor_attack(LABEL, [data])

#datasets = [
    #load_data("RNC1773_T-20_R25_A0002_R0050_G0_SNon_compressor_attack_2026-05-15_133812.json.gz"),
    #load_data("RNC1773_T-20_R25_A0006_R0050_G0_SNon_compressor_attack_2026-05-15_133836.json.gz"),
    #load_data("RNC1773_T-20_R25_A0020_R0050_G0_SNon_compressor_attack_2026-05-15_133904.json.gz"),
    #load_data("RNC1773_T-20_R25_A0060_R0050_G0_SNon_compressor_attack_2026-05-15_133946.json.gz"),
    #load_data("RNC1773_T-20_R25_A0200_R0050_G0_SNon_compressor_attack_2026-05-15_134008.json.gz"),
    #load_data("RNC1773_T-20_R25_A0600_R0050_G0_SNon_compressor_attack_2026-05-15_134027.json.gz"),
    #load_data("RNC1773_T-20_R25_A2000_R0050_G0_SNon_compressor_attack_2026-05-15_134045.json.gz"),
#]
#analyze_compressor_attack("RNC1773", datasets)


# RELEASE
# Recommended settings: ratio=high, attack=fast, Vstart=Vthres+20, Vstop=Vthres-20

#data = measure_compressor_release(LABEL, THRESHOLD_dBV + 20, THRESHOLD_dBV - 20)
#analyze_compressor_release(LABEL, [data])

#datasets = [
    #load_data("RNC1773_T-20_R25_A0002_R0050_G0_SNon_compressor_release_2026-05-15_134247.json.gz"),
    #load_data("RNC1773_T-20_R25_A0002_R0100_G0_SNon_compressor_release_2026-05-15_134323.json.gz"),
    #load_data("RNC1773_T-20_R25_A0002_R0300_G0_SNon_compressor_release_2026-05-15_134356.json.gz"),
    #load_data("RNC1773_T-20_R25_A0002_R0500_G0_SNon_compressor_release_2026-05-15_134432.json.gz"),
    #load_data("RNC1773_T-20_R25_A0002_R1000_G0_SNon_compressor_release_2026-05-15_134503.json.gz"),
    #load_data("RNC1773_T-20_R25_A0002_R3000_G0_SNon_compressor_release_2026-05-15_134538.json.gz"),
    #load_data("RNC1773_T-20_R25_A0002_R5000_G0_SNon_compressor_release_2026-05-15_134614.json.gz"),
#]
#analyze_compressor_release("RNC1773", datasets)

plt.show()
