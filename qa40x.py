import base64
import gzip
import json
import requests
import struct
import time

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

class Qa40x:
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
