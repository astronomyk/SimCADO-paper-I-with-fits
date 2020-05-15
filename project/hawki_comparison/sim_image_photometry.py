# * Use the magnitudes from Simbad to generate a Simcado object
# * Observe it with HAWKADO
# * Find the stars with photutils
# * Get the photometry

import numpy as np
from matplotlib import pyplot as plt

from copy import deepcopy
from scipy.signal import fftconvolve

from astropy import units as u
from astropy.io import fits, ascii
import simcado


def make_hawki_poppy_psf():
    import poppy

    osys = poppy.OpticalSystem()
    m1 = poppy.CircularAperture(radius=4.1)
    spiders = poppy.SecondaryObscuration(secondary_radius=0.6, n_supports=4,
                                         support_width=0.1)
    vlt = poppy.CompoundAnalyticOptic(opticslist=[m1, spiders], name='VLT')

    psfs = []
    seeing = 0.4
    see_psf = simcado.psf.seeing_psf(fwhm=seeing, size=384, pix_res=0.106 / 2)

    for lam in [1.2, 1.6, 2.2]:
        osys = poppy.OpticalSystem()
        osys.add_pupil(vlt)
        osys.add_detector(pixelscale=0.106, fov_arcsec=0.106 * 192)
        diff_psf = osys.calc_psf(lam * 1e-6)

        tmp = deepcopy(diff_psf)
        tmp[0].data = fftconvolve(diff_psf[0].data, see_psf[0].data,
                                  mode="same")
        tmp[0].data /= np.sum(tmp[0].data)

        tmp[0].header["SEEING"] = seeing
        tmp[0].header["CDELT1"] = tmp[0].header["PIXELSCL"]
        tmp[0].header["CDELT2"] = tmp[0].header["PIXELSCL"]

        psfs += tmp

    hdus = fits.HDUList(psfs)
    for i in range(1, len(hdus)):
        hdus[i] = fits.ImageHDU(data=hdus[i].data, header=hdus[i].header)

    hdus.writeto("HAWK-I_config/PSF_HAWKI_poppy.fits", clobber=True)

    fname = "HAWK-I_config/PSF_HAWKI_poppy.fits"

    plt.figure(figsize=(15,4))
    for i in range(3):
        plt.subplot(1,3,i+1)
        poppy.display_PSF(fname, ext=i, title="HAWKI PSF lambda="+str(fits.getheader(fname, ext=i)["WAVELEN"]))

    plt.show()


################################################################################
# Get the photon fluxes that should come out of simcado
################################################################################

def plot_background_flux():

    #K band 44  photons cm-2 s-1 A-1  - average BG is 13.6
    #H band 93  photons cm-2 s-1 A-1  - average BG is 14.4
    #J band 193 photons cm-2 s-1 A-1  - average BG is 16.5

    # This works out to ~ 1700 ph/pix/dit for J-band and ~12500 ph/pix/dit in K-band
    # Old calculation was with 4500 and 15000, but this was based on filter widths of 0.26 and 0.4um.
    # The HAWKI filters are more like 0.15 and 0.35um for J and Ks

    # Depending on which images we use, J BG flux ranges from 700 to 4000 and K from 12000 to 15000
    # This is exactly what we see in the images

    f0 = np.array([193, 93, 44]) * 1E4 * u.Unit("m-2 s-1 AA-1")
    area = (4.1**2 - 0.6**2)*np.pi * u.m**2
    dlam = np.array([1500, 2900, 3400]) * u.AA
    exptime = 10 * u.s
    mag = np.array([16.5, 14.4, 13.6])
    mag_corr = 10**(-0.4 * mag) / (u.arcsec**2)
    pix_size = 0.106**2 * u.arcsec**2
    inst_eff = 0.5
    gain = 1.7

    bg_ph_hawki = f0 * dlam * mag_corr * area * exptime * pix_size * inst_eff * gain
    print("HAWKI JHK Sky BG: [ADU/pix/dit]", bg_ph_hawki)


    ### Test the SimCADO background levels for HAWKI - rough calcluation

    sim_f0 = np.array([simcado.source.zero_magnitude_photon_flux(i) for i in ["J","H","Ks"]]) * u.Unit("ph/m2/s")
    print("BG flux JHK [ph/arcsec2/m2/s]", sim_f0 * mag_corr)
    print("BG flux JHK [ADU/pix/dit]", sim_f0 * mag_corr * area * exptime * pix_size * inst_eff * gain)
    print("Scale factor for system", area * exptime * pix_size * inst_eff * gain)


    # ## Create an optical train for the system
    # Use just the K=13.6 and J=16.5 and scaling a blank spectrum
    #
    # For K we get a background flux of ~18000 ph/pix/dit and J ~2500 ph/pix/dit. This is consistent with the upper and lower bounds set by the standard flux values given by theoretical values based on the ESO sky background values and empirical values from the HAWK-I ETC.
    #
    # The limits are  15000 < K < 28000 and 2000 < J < 2800 ph/pix/dit
    # This equates to 10000 < K < 16000 and 1200 < J < 1600 ADU/pix/dit


    cmd = simcado.UserCommands("HAWK-I_config/hawki.config")

    cmd["OBS_EXPTIME"] = 10
    cmd["INST_FILTER_TC"] = "J"
    cmd["ATMO_USE_ATMO_BG"] = "yes"
    cmd["ATMO_BG_MAGNITUDE"] = 16.5
    #cmd["ATMO_EC"] = None

    #cmd["SCOPE_USE_MIRROR_BG"] = "no"
    #cmd["FPA_USE_NOISE"] = "no"


    #cmd["FPA_LINEARITY_CURVE"] = None

    opt = simcado.OpticalTrain(cmd)
    fpa = simcado.Detector(cmd, small_fov=False)

    empty_sky = simcado.source.empty_sky()
    empty_sky.apply_optical_train(opt, fpa)

    myfits = fpa.read_out(OBS_EXPTIME=10)

    plt.imshow(myfits[0].data)
    plt.colorbar()
    plt.show()


################################################################################
# Ok lets make some stars and observe them
################################################################################

def simulate_m4_raw_data(show_plots=False):

    random_extinction_factor = 0.9
    k = simcado.optics.get_filter_curve("Ks") * random_extinction_factor

    cmd = simcado.UserCommands("HAWK-I_config/hawki.config")

    cmd["INST_FILTER_TC"] = k
    cmd["ATMO_BG_MAGNITUDE"] = 13.6
    cmd["OBS_EXPTIME"] = 10
    cmd["FPA_CHIP_LAYOUT"] = "HAWK-I_config/FPA_hawki_layout_cen.dat"
    cmd["FPA_LINEARITY_CURVE"] = "HAWK-I_config/FPA_hawki_linearity_ext.dat"

    opt = simcado.OpticalTrain(cmd)
    fpa = simcado.Detector(cmd, small_fov=False)

    if show_plots:
        plt.plot(opt.tc_source)
        plt.show()


    hdu_tbls = []
    for i in range(4):
        hdu_tbls += [ascii.read("M4/M4_chip"+str(i)+"table.dat")]

    gain = np.array([1.87, 1.735, 1.705, 2.11])
    hdus = []

    for i in range(4):
        x = (hdu_tbls[i]["x_corr"] - 1024) * 0.106
        y = (hdu_tbls[i]["y_corr"] - 1024) * 0.106

        k_m = hdu_tbls[i]["k_m"]

        src = simcado.source.stars(mags=k_m, x=x, y=y)
        fpa.chips[0].gain = gain[i]

        src.apply_optical_train(opt, fpa)
        hdus += fpa.read_out()

    for i in range(1, 4):
        hdus[i] = fits.ImageHDU(data=hdus[i].data, header=hdus[i].header)

    hdus = fits.HDUList(hdus)
    hdus.writeto("M4/hawkado_test10.fits", clobber=True)


simulate_m4_raw_data()