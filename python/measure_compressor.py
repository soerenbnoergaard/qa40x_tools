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
BUFFER_SIZE = 65536

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
        self.set_buffer_size(BUFFER_SIZE)
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
        assert(len(waveform_V) == BUFFER_SIZE)
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

def analyze_compressor_curve(label, datasets):
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

        ax.plot(Vi, G, ".-", color=f"C{idx}", label=label)
        ax.plot(Vi_comp, G_comp, "o", color=f"C{idx}", label=f"threshold={Vi_comp:.1f} dBV")

    ax.set_xlabel("Input [dBV]")
    ax.set_ylabel("Gain [dB]")
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(generate_filename(label, "compressor_curve", "png"))

def measure_compressor_attack(label, voltage_low_dBV=-40, voltage_high_dBV=-20, frequency_Hz=1000):
    qa = QA403()
    qa.set_buffer_size(BUFFER_SIZE)

    V_low = 10**(voltage_low_dBV/20.0)
    V_high = 10**(voltage_high_dBV/20.0)

    def generate_waveform():
        t = np.arange(BUFFER_SIZE) / SAMPLE_RATE_Hz
        waveform = np.exp(2j * np.pi * frequency_Hz * t)

        waveform[0:BUFFER_SIZE//2] *= V_low
        waveform[BUFFER_SIZE//2:] *= V_high
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
    dump_data(generate_filename(label, "compressor_attack"), data)
    return data

def analyze_compressor_attack(label, datasets):
    data = datasets[0]
    label = data["label"]
    f = data["frequency_Hz"]
    Vi = np.array(data["input_real_V"]) + 1j*np.array(data["input_imag_V"])    
    Vo = np.array(data["output_real_V"]) + 1j*np.array(data["output_imag_V"])    

    # Extract amplitude envelopes
    Ai = np.abs(Vi)
    Ao = np.abs(Vo)

    # Measurement points
    n_low = BUFFER_SIZE//4
    n_high = 7*BUFFER_SIZE//8

    # Measure small signal gain
    G = Ao[n_low] / Ai[n_low]
    print(f"Small signal gain = {20*np.log10(G)} dB")
    Ao_exp = G * Ai[n_high]
    print(f"Expected output voltage = {20*np.log10(Ao_exp):.1f} dBV")
    Ao_settled = Ao[n_high]
    print(f"Settled output voltage = {20*np.log10(Ao_settled):.1f} dBV")
    print(f"Gain reduction = {20*np.log10(Ao_exp/Ao_settled):.1f} dB")

    # Measure attack time as: Time from transient until 90% of the gain reduction is effective
    n0 = BUFFER_SIZE//2
    GR = Ao_exp - Ao_settled
    Ao_90 = Ao_settled + (0.1 * GR)
    n1 = n0
    for n in reversed(range(n0, n_high)):
        if Ao[n] > Ao_90:
            n1 = n
            break
    n_attack = n1 - n0
    t_attack = n_attack / SAMPLE_RATE_Hz
    print(f"Attack time = {t_attack * 1e3:.3f} ms")

    fig, ax = plt.subplots()
    ax.plot(Ai)
    ax.plot(Ao)
    ax.axhline(Ao_exp, color="k")
    ax.axhline(Ao_settled, color="k")
    ax.plot([n1], [Ao_90], "ok")
    fig.tight_layout()
    fig.savefig(generate_filename(label, "compressor_attack", "png"))


LABEL = "fmr_rnc1773"

#data_ref = load_data("fmr_rnc1773_compressor_curve_2026-05-15_102808.json")
#data = measure_compressor_curve(LABEL, -50, 0, 5)
#analyze_compressor_curve(LABEL, [data_ref, data])

#data = measure_compressor_attack(LABEL, -50, -10)
data = load_data("fmr_rnc1773_compressor_attack_2026-05-15_104232.json.gz")
analyze_compressor_attack(LABEL, [data])
plt.show()
