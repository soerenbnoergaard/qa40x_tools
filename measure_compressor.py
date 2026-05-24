import time
import numpy as np
import matplotlib.pyplot as plt
import qa40x as qa
import analyze_compressor as analyze

def measure_compressor_curve(label, amplitude_min_dBV, amplitude_max_dBV, amplitude_step_dB, frequency_Hz=1000):
    assert (
        (amplitude_step_dB > 0 and amplitude_min_dBV < amplitude_max_dBV) or
        (amplitude_step_dB < 0 and amplitude_min_dBV > amplitude_max_dBV)
    )
    inst = qa.Qa40x()
    inputs_dBV = []
    outputs_dBV = []

    input_dBV = amplitude_min_dBV
    while input_dBV <= amplitude_max_dBV:
        inst.setup_gen1(True, frequency_Hz, input_dBV)
        inst.acquire()
        output_dBV = float(inst.measure_rms_dBV())
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
    qa.dump_data(qa.generate_filename(label, "compressor_curve"), data)
    return data

def measure_compressor_attack_release(is_release, label, voltage_start_dBV=-40, voltage_stop_dBV=-20, frequency_Hz=1000):
    measurement_label = "compressor_release" if is_release else "compressor_attack"
    inst = qa.Qa40x()

    if is_release:
        buffer_size = 262144
    else:
        buffer_size = 65536

    inst.set_buffer_size(buffer_size)

    V_start = 10**(voltage_start_dBV/20.0)
    V_stop = 10**(voltage_stop_dBV/20.0)

    def generate_waveform():
        t = np.arange(buffer_size) / qa.SAMPLE_RATE_Hz
        waveform = np.exp(2j * np.pi * frequency_Hz * t)

        n_transient = 1*buffer_size//8
        waveform[0:n_transient] *= V_start
        waveform[n_transient:] *= V_stop
        return waveform

    Vi = generate_waveform()

    inst.acquire_custom_waveform(np.real(Vi))
    Vo_i = np.array(inst.measure_recorded_waveform()["Left"])
    time.sleep(3)
    inst.acquire_custom_waveform(np.imag(Vi))
    Vo_q = np.array(inst.measure_recorded_waveform()["Left"])
    Vo = Vo_i + 1j*Vo_q

    data = {
        "label": label,
        "frequency_Hz": frequency_Hz,
        "input_real_V": np.real(Vi).tolist(),
        "input_imag_V": np.imag(Vi).tolist(),
        "output_real_V": np.real(Vo).tolist(),
        "output_imag_V": np.imag(Vo).tolist(),
    }
    qa.dump_data(qa.generate_filename(label, measurement_label), data)
    return data

def measure_compressor_attack(*args, **kwargs):
    return measure_compressor_attack_release(False, *args, **kwargs)

def measure_compressor_release(*args, **kwargs):
    return measure_compressor_attack_release(True, *args, **kwargs)


if __name__ == "__main__":
    LABEL = "eureka_line_T-32_A00_RA00_RE10_G0_SOFToff"
    THRESHOLD_dBV = -40


    # CURVE
    # Recommended settings: attack=fast, release=fast

    # data = measure_compressor_curve(LABEL, THRESHOLD_dBV - 10, THRESHOLD_dBV + 10, 1)
    # analyze.analyze_compressor_curve(LABEL, [data])


    # ATTACK
    # Recommended settings: ratio=high, release=fast, Vstart=Vthres-20, Vstop=Vthres+20

    # data = measure_compressor_attack(LABEL, THRESHOLD_dBV - 20, THRESHOLD_dBV + 20)
    # analyze.analyze_compressor_attack(LABEL, [data])


    # RELEASE
    # Recommended settings: ratio=high, attack=fast, Vstart=Vthres+20, Vstop=Vthres-20

    data = measure_compressor_release(LABEL, THRESHOLD_dBV + 20, THRESHOLD_dBV - 20)
    analyze.analyze_compressor_release(LABEL, [data])


    plt.show()
