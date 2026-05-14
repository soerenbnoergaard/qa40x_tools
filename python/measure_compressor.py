import time
import requests
import struct
import base64

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

def save_records(records, filename_base):
    filename = filename_base + "_" + time.strftime("%Y-%m-%d_%H%M%S") + ".csv"
    keys = records[0].keys()
    header = ",".join(keys)
    with open(filename, "w") as fh:
        fh.write(header + "\n")
        for record in records:
            line = ",".join([str(record[k]) for k in keys])
            fh.write(line + "\n")
    print(f"Write to {filename} complete!")

def load_records(filename):
    df = pd.read_csv(filename)
    return df.to_records(index=False)

def lerp(x, x0, y0, x1, y1):
    return (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)

def measure_compressor_curve(amplitude_min_dBV, amplitude_max_dBV, amplitude_step_dB, frequency_Hz=1000):
    assert (
        (amplitude_step_dB > 0 and amplitude_min_dBV < amplitude_max_dBV) or
        (amplitude_step_dB < 0 and amplitude_min_dBV > amplitude_max_dBV)
    )
    qa = QA403()
    records = []

    input_dBV = amplitude_min_dBV
    while input_dBV <= amplitude_max_dBV:
        qa.setup_gen1(True, frequency_Hz, input_dBV)
        qa.acquire()
        output_dBV = qa.measure_rms_dBV()
        record = {
            "frequency_Hz": float(frequency_Hz),
            "input_dBV": float(input_dBV),
            "output_dBV": float(output_dBV),
        }
        print(record)
        records.append(record)
        input_dBV += amplitude_step_dB

    return records

def analyze_compressor_curve(records):
    df = pd.DataFrame.from_records(records)
    Vi = df["input_dBV"]
    Vo = df["output_dBV"]
    G = Vo - Vi

    # Find threshold as the 1 dB compression point
    G_comp = G[0] - 1
    Vi_comp = np.nan
    for n in range(1, len(G)):
        if G[n] < G_comp:
            Vi_comp = lerp(G_comp, G[n-1], Vi[n-1], G[n], Vi[n])

    print(f"Threshold: Vi={Vi_comp:.1f} dBV, Vo={Vi_comp+G_comp:.1f} dBV, G={G_comp:.1f} dB")

    fig, ax = plt.subplots()
    ax.plot(Vi, G, ".-")
    ax.plot(Vi_comp, G_comp, "o")
    ax.set_xlabel("Input [dBV]")
    ax.set_ylabel("Gain [dB]")
    fig.tight_layout()

def measure_compressor_attack(voltage_low_dBV=-40, voltage_high_dBV=-20, frequency_Hz=1000):
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
    qa.acquire_custom_waveform(np.imag(Vi))
    Vo_q = np.array(qa.measure_recorded_waveform())
    Vo = Vo_i + 1j*Vo_q

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

#records = measure_compressor_curve(-50, 0, 5)
#save_records(records, "fmr_rnc1773")
#records = load_records("")
#analyze_compressor_curve(records)

measure_compressor_attack(-50, -10)
plt.show()
