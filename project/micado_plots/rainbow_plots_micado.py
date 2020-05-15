import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import simcado
from project.etc import ETC


def plot_micado_rainbow(ab_mags=False):
    axs = [[0.1, 0.5, 0.4, 0.4],
           [0.5, 0.5, 0.4, 0.4],
           [0.1, 0.1, 0.4, 0.4],
           [0.5, 0.1, 0.4, 0.4],
           [0.1, 0.9, 0.8, 0.03]]

    text_heights = [np.array([28.3, 27.2, 25.7, 24.6, 22.9, 21.4])+1.6,
                    np.array([28.2, 27.2, 25.7, 24.6, 22.9, 21.3])+1,
                    np.array([28.3, 27.2, 25.7, 24.6, 22.9, 21.4])+0.3,
                    np.array([28.3, 27.2, 25.7, 24.6, 22.9, 21.3])-0.9]

    mmins = [19, 19, 18, 16]

    filter_names = ["J", "H", "Ks", "Br-gamma"]
    label_names = ["J", "H", "K$_S$", "Br$\gamma$"]

    ab = [0.9, 1.4, 1.85, 1.85] if ab_mags else [0] * 4
    bg_mags = [16.5, 14.4, 13.6, 13.6]

    exptimes = np.logspace(np.log10(60), np.log10(10*3600), 10)

    plt.figure(figsize=(12, 8))

    for i in range(4):
        filt_name = filter_names[i]
        cmd = simcado.UserCommands()
        cmd["FPA_LINEARITY_CURVE"] = None
        cmd["ATMO_BG_MAGNITUDE"] = 16.5
        cmd["INST_FILTER_TC"] = filt_name

        opt = simcado.OpticalTrain(cmd)
        micado_etc = ETC(cmds=cmd, opt_train=opt, mmin=mmins[i], mmax=32)

        map_ax = plt.axes(axs[i])

        snrs = micado_etc.snr(exptimes=exptimes, fitted=True)
        micado_etc.plot_snr_rainbow(exptimes/3600, snrs,
                                    snr_levels=[1, 5, 10, 50, 250, 1000],
                                    text_heights=text_heights[i] + ab[i],
                                    text_center=5,
                                    use_colorbar=False)

        plt.ylim(18, 31.9)
        plt.xlim(0, 9.9)
        plt.ylabel(label_names[i], fontsize=14)
        if i in [1, 3]:
            plt.gca().yaxis.tick_right()
            plt.gca().yaxis.set_label_position("right")
        if i in [0, 1]:
            plt.gca().set_xticklabels("")

    plt.axes(axs[4])
    cb = plt.colorbar(cax=plt.gca(), orientation="horizontal")
    plt.gca().xaxis.tick_top()

    plt.text(0, 48, "Signal to Noise Ratio ($\sigma$)",
             horizontalalignment="center", fontsize=14)
    plt.text(0, 16, "Exposure time [hours]", horizontalalignment="center",
             fontsize=14)
    if ab_mags:
        plt.text(-11.5, 32, "AB magnitudes", rotation=90,
                 verticalalignment="center", fontsize=14)
    else:
        plt.text(-11.5, 32, "Vega magnitudes", rotation=90,
                 verticalalignment="center", fontsize=14)

    plt.savefig("images/MICADO_SNR_Rainbow_JHKBrG_ab.png", format="png")
    plt.savefig("images/MICADO_SNR_Rainbow_JHKBrG_ab.pdf", format="pdf")


def show_limiting_mags_micado():
    filter_names = ["J", "H", "Ks", "Br-gamma"]
    bg_mags = [16.5, 14.4, 13.6, 13.6]
    exptimes = np.array([2.6, 10, 60, 600, 3600, 18000, 36000])
    mmins = [19, 19, 18, 16]

    etcs = []

    for i in range(4):
        cmd = simcado.UserCommands()
        cmd["FPA_LINEARITY_CURVE"] = None
        cmd["ATMO_BG_MAGNITUDE"] = bg_mags[i]
        cmd["INST_FILTER_TC"] = filter_names[i]

        opt = simcado.OpticalTrain(cmd)
        etcs += [ETC(cmds=cmd, opt_train=opt, mmin=mmins[i], mmax=30)]

    plt.figure(figsize=(15, 15))

    my_lim_mags = []
    for i in range(4):
        plt.subplot(2, 2, i + 1)

        my_snr = etcs[i]

        lim_mags = my_snr.limiting_magnitudes(exptimes=exptimes, plot=True,
                                              limiting_sigmas=[5, 10, 250])  #
        lim_mags = np.array(lim_mags).T

        my_lim_mags += [lim_mags]

        for lm, sig in zip(lim_mags, [5, 10, 250]):
            print(str(filter_names[i]) + " & " + str(sig) + r"\sig & " +
                  str([str(np.round(m, 1))[:4] + r"\m  &  " for m in lm])[
                  1:-1].replace(",", "").replace("'", "") + r"\\")

    plt.savefig("images/MICADO_limiting_mags_JHKBrG.png", format="png")
    plt.savefig("images/MICADO_limiting_mags_JHKBrG.pdf", format="pdf")

    return my_lim_mags


plot_micado_rainbow(True)
# lim_mags = show_limiting_mags_micado()
# print(lim_mags)
