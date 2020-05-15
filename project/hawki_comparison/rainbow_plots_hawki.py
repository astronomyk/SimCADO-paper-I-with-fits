import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import simcado
from project.etc import ETC

filt_name = "J"
cmd = simcado.UserCommands("HAWK-I_config/hawki.config")
cmd["FPA_LINEARITY_CURVE"] = None
cmd["ATMO_BG_MAGNITUDE"] = 16.5
cmd["INST_FILTER_TC"] = filt_name

opt = simcado.OpticalTrain(cmd)
hawki_etc = ETC(cmds=cmd, opt_train=opt, mmin=16, mmax=26)

exptimes = np.logspace(0.3, 3.55, 10)
snrs = hawki_etc.snr(exptimes=exptimes, fitted=True)
hawki_etc.plot_snr_rainbow(exptimes/60, snrs, snr_levels=[1, 5, 10, 50, 250, 1000],
                           text_heights=np.array([21.3,20,18.8,17.5,15.8,14.3])+3.5,
                           text_center=30)

plt.savefig("images/HAWKI_rainbow_J.png", format="png")
plt.savefig("images/HAWKI_rainbow_J.pdf", format="pdf")
