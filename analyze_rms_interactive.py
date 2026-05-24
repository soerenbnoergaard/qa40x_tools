import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn.objects as so

plt.style.use("seaborn-v0_8-darkgrid")

def mean_of_dBV(ser):
    return 20*np.log10(np.mean(10**(ser/20)))

def analyze(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    g = (
        so.Plot(
            data=df,
            x="source_impedance_ohm",
            y="input_noise_dBV",
            color="label",
        )
        # .add(so.Dots())
        .add(so.Line(marker="o"), so.Agg(mean_of_dBV))
        .scale(x="log")
        .on(ax)
        .plot()
    )

    legend = fig.legends.pop(0)
    ax.legend(legend.legend_handles, [t.get_text() for t in legend.texts], loc="upper left")
    fig.tight_layout()

def load(filename, gain_dB, label):
    return (
        pd.read_csv(filename, names=("source_impedance_ohm", "output_noise_dBV"))
        .assign(label=label)
        .eval("input_noise_dBV = output_noise_dBV - @gain_dB")
    )


df = pd.concat([
    load("001_fethead_into_digimax_gain_50.5dB.csv", 50.5, "Fethead + Digimax FS"),
    load("002_matchbox_into_fethead_into_digimax_gain_59.8dB.csv", 59.8, "Matchbox + Fethead + Digimax FS"),
    load("003_presonus_digimax_gain_61.5dB.csv", 61.5, "Digimax FS"),
    load("004_gap_preq-73_gain_82.6dB.csv", 82.6, "GAP PREQ-73"),
    load("005_babyface_pro_gain_81.4dB.csv", 81.4, "Babyface Pro"),
], ignore_index=True)

analyze(df)

plt.savefig("python/interactive_log_rms_analyze.png")
plt.show()