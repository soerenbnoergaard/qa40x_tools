import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import qa40x as qa

def measure(label=""):
    inst = qa.Qa40x()
    inst.acquire()
    fodata = inst.measure_played_spectrum()
    todata = inst.measure_played_waveform()
    fidata = inst.measure_recorded_spectrum()
    tidata = inst.measure_recorded_waveform()

    data = {
        "fodata": fodata,
        "tidata": tidata,
        "todata": todata,
        "fidata": fidata,
    }
    qa.dump_data(qa.generate_filename(label, "impulse_response"), data)
    return data

def analyze(data):
    t = data["tidata"]["Time"]
    x = data["todata"]["Left"]
    y = data["tidata"]["Left"]

    f = data["fidata"]["Frequency"]
    X = data["fodata"]["Left"]
    Y = data["fidata"]["Left"]

    fig, ax = plt.subplots()
    ax.plot(t, x)
    ax.plot(t, y)

    # TODO: Deconvolution + compare with the extracted output spectrum from Qa40x
    # https://hal.science/hal-02504321v1/document
    # https://www.researchgate.net/publication/237724101_DIRECTOR%27S_CUT_INCLUDING_PREVIOUSLY_UNRELEASED_MATERIAL_AND_SOME_CORRECTIONS
    # https://pyfar-gallery.readthedocs.io/en/latest/gallery/no_binder/impulse_response_measurement.html
    # https://www.carminecella.com/teaching/IR_calc.pdf

    fig, ax = plt.subplots()
    ax.semilogx(f, 20 * np.ma.log10(X))
    ax.semilogx(f, 20 * np.ma.log10(Y))
    ax.set_xlim(20, 20e3)
    ax.set_ylim(-170, 20)


if __name__ == "__main__":
    # data = measure()
    # analyze(data)

    analyze(qa.load_data("_impulse_response_2026-05-24_123051.json.gz"))
    plt.show()
