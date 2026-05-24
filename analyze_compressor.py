import glob
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import qa40x as qa

plt.style.use("bmh")

def lerp(x, x0, y0, x1, y1):
    return (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)

def rsquared(x, y):
    corr_matrix = np.corrcoef(x, y)
    corr = corr_matrix[0, 1]
    return corr ** 2

def calculate_threshold(Vi, G):
    # Find threshold as the 1 dB compression point
    G_comp = G[0] - 1
    Vi_comp = np.nan
    for n in range(1, len(G)):
        if G[n] < G_comp:
            Vi_comp = lerp(G_comp, G[n-1], Vi[n-1], G[n], Vi[n])
            return SimpleNamespace(
                Vi_comp=Vi_comp,
                G_comp=G_comp,
                n_pre=n-1,
                n_post=n,
            )
    return SimpleNamespace(
        Vi_comp=Vi_comp,
        G_comp=G_comp,
        n_pre=len(G)-3,
        n_post=len(G)-2,
    )

def calculate_ratio(Vi, G):
    # Input data is the curve above the threshold

    m, b = np.polyfit(Vi, G, 1)
    r2 = rsquared(Vi, G)

    # The ratio is reciprocal of the slope of the Vo vs Vi curve
    ratio = 1 / (m + 1)

    return SimpleNamespace(
        ratio=ratio,
        slope=m,
        yintersect=b,
        r2=r2,
    )

def analyze_compressor_curve(super_label, datasets, gain_curve=False):
    fig, ax = plt.subplots()
    for idx, data in enumerate(datasets):
        label = data["label"]
        Vi = np.array(data["input_dBV"])
        Vi = np.array(data["input_dBV"])
        Vo = np.array(data["output_dBV"])
        G = Vo - Vi

        threshold_data = calculate_threshold(Vi, G)
        ratio_data = calculate_ratio(Vi[threshold_data.n_post:], G[threshold_data.n_post:])
        gain_fit = lambda _Vi: ratio_data.slope * _Vi + ratio_data.yintersect
        io_fit = lambda _Vi: (ratio_data.slope + 1) * _Vi + ratio_data.yintersect

        print("{}: Vi_comp={:.1f} dBV, Vo_comp={:.1f} dBV, G_comp={:.1f} dB, ratio={:.1f}:1, slope={:.3f}, R-squared={:.3f}".format(
            label,
            threshold_data.Vi_comp,
            threshold_data.Vi_comp + threshold_data.G_comp,
            threshold_data.G_comp,
            ratio_data.ratio,
            ratio_data.slope,
            ratio_data.r2,
        ))

        color = f"C{idx}"
        legend_label = "{}\nthreshold={:.1f} dBV, ratio={:.1f}:1".format(
            label,
            threshold_data.Vi_comp,
            ratio_data.ratio,
        )

        if gain_curve:
            ax.plot(Vi, G, ".-", color=color, label=legend_label)
            ax.plot(threshold_data.Vi_comp, threshold_data.G_comp, "o", color=color)
            # ax.plot([Vi[0], Vi[-1]], [gain_fit(Vi[0]), gain_fit(Vi[-1])], color=color, linestyle="--")

        else:
            ax.plot(Vi, Vo, ".-", color=color, label=legend_label)
            ax.plot(threshold_data.Vi_comp, threshold_data.G_comp + threshold_data.Vi_comp, "o", color=color)
            # ax.plot([Vi[0], Vi[-1]], [io_fit(Vi[0]), io_fit(Vi[-1])], color=color, linestyle="--")

    ax.set_xlabel("Input [dBV]")
    ax.set_ylabel("Gain [dB]" if gain_curve else "Output [dBV]")
    ax.legend(loc="lower left" if gain_curve else "upper left", ncols=1, fontsize=6)
    fig.tight_layout()
    fig.savefig(qa.generate_filename(super_label, "compressor_curve", "png"))

def _analyze_compressor_attack_release(is_release, super_label, datasets, use_hilbert=False):
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
        t_dur = n_dur / qa.SAMPLE_RATE_Hz
        if is_release:
            search_label = f"release={t_dur * 1e3:.3f} ms"
        else:
            search_label = f"attack={t_dur * 1e3:.3f} ms"
        print(f"Measured {search_label}")

        color = f"C{idx}"
        f = lambda x: 20*np.log10(np.abs(x))

        t = (np.arange(len(Ao)) - n_transient) / qa.SAMPLE_RATE_Hz * 1000
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

    ax.legend(loc="lower right" if is_release else "upper right", ncols=1, fontsize=6)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Output amplitude [dBV]")
    ax.set_ylim(ymin - 3, ymax + 3)
    fig.tight_layout()
    fig.savefig(qa.generate_filename(super_label, measurement_label, "png"))

def analyze_compressor_attack(*args, **kwargs):
    return _analyze_compressor_attack_release(False, *args, **kwargs)

def analyze_compressor_release(*args, **kwargs):
    return _analyze_compressor_attack_release(True, *args, **kwargs)

if __name__ == "__main__":
    # CURVE
    datasets = [qa.load_data(_f) for _f in sorted(glob.glob("*compressor_curve_*.json.gz"))]
    analyze_compressor_curve("", datasets)

    # ATTACK
    datasets = [qa.load_data(_f) for _f in sorted(glob.glob("*compressor_attack_*.json.gz"))]
    analyze_compressor_attack("", datasets)

    # RELEASE
    datasets = [qa.load_data(_f) for _f in sorted(glob.glob("*compressor_release_*.json.gz"))]
    analyze_compressor_release("", datasets)

    plt.show()
