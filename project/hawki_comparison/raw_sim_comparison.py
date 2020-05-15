# * Compare the two photometries
# * Plot the two images next to each other
# * Plot the photometries for J and Ks
# * Generate the limiting magnitude vs time graph for HAWKI with the SNR contours

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats

from astropy import units as u
from astropy.io import fits, ascii
from astropy import wcs
from astropy.table import Table, Column
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
from photutils import find_peaks
from photutils import CircularAnnulus, aperture_photometry

from astropy.coordinates import SkyCoord

import simcado

#Simbad.ROW_LIMIT = 99999

from copy import deepcopy
from scipy.signal import fftconvolve

################################################################################
# Compare photometry of raw vs sim'ed data
################################################################################

def combine_photometry_tables():
    hdu_tbls = []
    for i in range(4):
        hdu_tbls += [ascii.read("M4/M4_chip"+str(i)+"table.dat")]

    for i in range(4):
        fluxes = []

        hdu = fits.open("M4/hawkado_test10.fits")[i]
        im = hdu.data

        dp = 10

        tbl = hdu_tbls[i]

        for j in range(len(tbl)):

            x = int(tbl["x_corr"][j])
            y = int(tbl["y_corr"][j])

            im2 = im[y-dp:y+dp+1, x-dp:x+dp+1]
            #mean, median, std = sigma_clipped_stats(im2, sigma=3.0, iters=3)
            median = np.median(im2)
            im2 -= median

            try:
                raise ValueError
                flux = get_flux(im2)
                fluxes += [flux]
            except:
                q = 9
                flux = np.sum(im2[dp-q:dp+q+1,dp-q:dp+q+1])
                fluxes += [flux]

            if j%100==0:
                print(j, len(tbl))

        if "flux_sim" in hdu_tbls[i].colnames:
            hdu_tbls[i].remove_column('flux_sim')
        col = Column(data=np.nan_to_num(fluxes), name="flux_sim")
        hdu_tbls[i].add_column(col)

    return hdu_tbls


def make_plot_hawki_vs_hawkado_counts(hdu_tbls):

    qual_dict = {"A": 1, "B": 0.6, 'C': 0.4, 'D': 0.2, 'E': 0, 'U': 0}
    x = np.array([1.1E4, 1.1E8])

    plt.figure(figsize=(10, 10))

    for i, c in zip([1, 0, 3, 2], "rgbk"):
        plt.subplot(2, 2, i + 1)

        q = [q[2] for q in hdu_tbls[i]["ph_qual"]]
        for z in "ABCDEU":
            mask = [w == z for w in q]
            adu_hawki = hdu_tbls[i]["flux"][mask]
            adu_hawkado = hdu_tbls[i]["flux_sim"][mask]
            plt.plot(adu_hawki, adu_hawkado, c + "+", alpha=qual_dict[z])

        ratio = adu_hawkado / adu_hawki - 1
        ratio = ratio[(ratio < 5)]
        print(np.median(ratio), np.std(ratio))

        plt.plot(x, 1.3 * x, "k--", alpha=0.5)
        plt.plot(x, x, "k", alpha=0.5)
        plt.plot(1.3 * x, x, "k--", alpha=0.5)
        plt.legend(["Chip " + str(i + 1)], loc=2)
        plt.loglog()

        x0, x1 = 2E4, 1.1E7
        x2, x3 = list(25.8 - 2.5 * np.log10((x0, x1)))

        plt.xlim(x0, x1);
        plt.ylim(x0, x1)

        if i in [0, 1]:
            plt.xticks([])
            plt.twiny()
            plt.xlim(x2, x3)
        if i in [1, 3]:
            plt.yticks([])
            plt.twinx()
            plt.ylim(x2, x3)

    plt.text(0.8E4, 1E7, "Simulated Image Aperture counts [ADU]", rotation=90,
             verticalalignment="center")
    plt.text(1E7, 0.8E4, "HAWKI Raw Image Aperture counts [ADU]",
             horizontalalignment="center")

    plt.text(1.2E10, 1E7, "Simulated Image K-band Magnitudes (zp=25.8)",
             rotation=270, verticalalignment="center")
    plt.text(1E7, 1.2E10, "HAWKI Raw Image K-band Magnitudes (zp=25.8)",
             horizontalalignment="center")

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig("images/HAWKI_vs_HAWKado_counts.png", format="png")
    plt.savefig("images/HAWKI_vs_HAWKado_counts.pdf", format="pdf")

    ############################################################################
    # make ratio plot
    ############################################################################

    qual_dict = {"A": 1, "B": 0.6, 'C': 0.4, 'D': 0.2, 'E': 0, 'U': 0}
    x = np.array([1.1E4, 1.1E8])

    plt.figure(figsize=(10, 3))

    hists = []
    for i, c in zip([1, 0, 3, 2], "rgbk"):
        plt.subplot(2, 2, i + 1)

        q = [q[2] for q in hdu_tbls[i]["ph_qual"]]
        for z in "ABCDEU":
            mask = [w == z for w in q]
            adu_hawki = hdu_tbls[i]["flux"][mask]
            adu_hawkado = hdu_tbls[i]["flux_sim"][mask]
            plt.plot(adu_hawkado, adu_hawki / adu_hawkado - 1, c + "+",
                     alpha=qual_dict[z])

            if z == "A":
                ratio = hdu_tbls[i]["flux"] / hdu_tbls[i]["flux_sim"] - 1
                ratio = ratio[(ratio < 5) * (ratio > -5)]
                hists += [np.histogram(ratio, bins=np.linspace(-2, 2, 41))]

        y = 0.25
        plt.plot(x, [y, y], "k--", alpha=0.5)
        plt.plot(x, [0, 0], "k--", alpha=0.5)
        plt.plot(x, [-y, -y], "k--", alpha=0.5)
        plt.semilogx()

        x0, x1 = 2E4, 1.1E7
        x2, x3 = list(25.8 - 2.5 * np.log10((x0, x1)))

        plt.xlim(x0, x1);
        plt.ylim(-1, 0.99)

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig("images/HAWKI_vs_HAWKado_ratios.png", format="png")
    plt.savefig("images/HAWKI_vs_HAWKado_ratios.pdf", format="pdf")


def plot_2mass_vs_hawkado_inst_mags(hdu_tbls):

    zero_point = 28.1
    qual_dict = {"A": 1, "B": 0.6, 'C': 0.4, 'D': 0.2, 'E': 0, 'U': 0}

    plt.figure(figsize=(5, 5))

    a = []
    for i, c in zip([0, 1, 2, 3], ["ro", "g^", "bs", "y+"]):

        q = [q[2] for q in hdu_tbls[i]["ph_qual"]]
        for z in "ABCDEU":
            mask = [w == z for w in q]
            mag_2mass = hdu_tbls[i]["k_m"][mask]
            mag_hawkado = zero_point - 2.5 * np.log10(hdu_tbls[i]["flux_sim"][mask])
            b = plt.scatter(mag_2mass, mag_hawkado, c=c[0], marker=c[1],
                            alpha=qual_dict[z] * 0.7, s=10, edgecolor="none")
            if z == "A":
                a += [b]

        plt.plot([8, 18], [8, 18], "k--", alpha=0.5)

    plt.legend(a, ["Chip 1", "Chip 2", "Chip 3", "Chip 4"], loc=2)
    plt.xlim(7.8, 16.9)
    plt.ylim(7.8, 16.9)

    plt.text(6.5, 12.5, "Simulated HAWK-I K-band Magnitudes [mag]", rotation=90,
             verticalalignment="center")
    plt.text(12.5, 7, "2MASS K-band Magnitudes [mag]", horizontalalignment="center")

    plt.savefig("images/2MASS_vs_HAWKado_inst_mags_single.png", format="png")
    plt.savefig("images/2MASS_vs_HAWKado_inst_mags_single.pdf", format="pdf")


def plot_image_comparison():
    im_hawkado = fits.getdata("M4/hawkado_test10.fits", ext=2)
    im_hawki = fits.getdata("M4/HAWKI.2015-06-10T05_12_28.683.fits")

    plt.figure(figsize=(12, 9))

    ########################### HAWKI
    plt.axes([0.05, 0.3, 0.45, 0.6])
    ax_hawki = plt.imshow(im_hawki[1024:, 1024:], norm=LogNorm(), vmin=1.3E4,
                          vmax=3E4, origin="lower",
                          interpolation="none", cmap="Greys", aspect="auto")
    plt.xticks([])
    plt.yticks([])
    plt.text(500, 1200, "Raw HAWKI image of M4", fontsize=16,
             horizontalalignment="center")

    ########################### HAWKado
    plt.axes([0.5, 0.3, 0.45, 0.6])
    ax_hawkado = plt.imshow(im_hawkado[1024:, 1024:], norm=LogNorm(), vmin=1E4,
                            vmax=2.7E4, origin="lower",
                            interpolation="none", cmap="Greys", aspect="auto")
    plt.xticks([])
    plt.yticks([])
    plt.text(500, 1200, "Simulated HAWKI image of M4", fontsize=16,
             horizontalalignment="center")


    ########################### HAWKI Hist
    plt.axes([0.5, 0.05, 0.45, 0.25])
    plt.hist(im_hawkado[1024:, 1024:].flatten(), bins=np.logspace(3, 6, 30))

    plt.loglog()
    plt.yscale('log', nonposy='clip')
    plt.xlim(8E2, 8E5)
    plt.ylim(1, 5E6)
    plt.gca().yaxis.tick_right()

    ########################### HAWKado Hist
    plt.axes([0.05, 0.05, 0.45, 0.25])
    plt.hist(im_hawki[1024:, 1024:].flatten(), bins=np.logspace(3, 6, 30))

    plt.loglog()
    plt.yscale('log', nonposy='clip')
    plt.xlim(8E2, 8E5)
    plt.ylim(1, 5E6)

    plt.text(1E6, 1E-2, "Pixel values [ADU]", horizontalalignment="center",
             fontsize=11)
    plt.text(4E2, 1E3, "# Pixels", rotation=90, verticalalignment="center",
             fontsize=11)


    ########################### CB
    ax = plt.axes([0.05, 0.9, 0.9, 0.05])
    ticks = [15000, 25000]
    cb = plt.colorbar(mappable=ax_hawki, cax=ax, ticks=ticks,
                      orientation="horizontal")
    ax.xaxis.tick_top()
    from matplotlib.ticker import ScalarFormatter
    ax.set_ticklabel_format = ScalarFormatter()

    plt.savefig("images/HAWKI_vs_HAWKado_images_quarter.png", format="png")
    plt.savefig("images/HAWKI_vs_HAWKado_images_quarter.pdf", format="pdf")


hdu_tbls = combine_photometry_tables()
# make_plot_hawki_vs_hawkado_counts(hdu_tbls)
plot_2mass_vs_hawkado_inst_mags(hdu_tbls)
#plot_image_comparison()
