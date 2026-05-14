import time
import requests

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class QA403:
    def post(self, s):
        return requests.post(f"http://localhost:9402/{s}").json()

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


records = measure_compressor_curve(-60, -20, 1)
save_records(records, "presonus_eureka")
#records = load_records("presonus_eureka_2026-05-14_135850.csv")
#analyze_compressor_curve(records)
#records = load_records("presonus_eureka_2026-05-14_140246.csv")
#analyze_compressor_curve(records)
plt.show()
