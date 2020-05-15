# * Import a fits file image,
# * Load in the stars from astroquery
# * Find stars with photutils
# * Shift the image to match the photometry
# * Get the photometry

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

from astropy.coordinates import SkyCoord

import simcado

#Simbad.ROW_LIMIT = 99999

################################################################################
# Read in the image and pull in the 2mass catalogue
################################################################################

#global_fname = "NGC4147/HAWKI.2014-01-19T07_49_48.826.fits"
#gloabl_tbl_name = "NGC4147/fp_pscNGC4147.tbl"

global_fname = "M4/HAWKI.2015-06-10T05_12_28.683.fits"
gloabl_tbl_name = "M4/fp_2mass.fp_psc9134.tbl.txt"

f = fits.open(global_fname)[3]
my_wcs = wcs.WCS(f.header)
oc_2mass = ascii.read(gloabl_tbl_name)   # Ks band

################################################################################
# Display one of the images and plot the catalogue coordinates
################################################################################

arr_coords_2mass = np.array((oc_2mass["ra"], oc_2mass["dec"])).T
arr_pixels_2mass = my_wcs.wcs_world2pix(arr_coords_2mass, 1)

mask_2mass = (arr_pixels_2mass[:,0] > 0) * (arr_pixels_2mass[:,0] < 2048) * \
             (arr_pixels_2mass[:,1] > 0) * (arr_pixels_2mass[:,1] < 2048)

fig = plt.figure(figsize=(12, 8))
fig.add_subplot(111)#, projection=my_wcs)

plt.scatter(arr_pixels_2mass[:,0][mask_2mass], arr_pixels_2mass[:,1][mask_2mass],
            facecolors='none', edgecolors='r')

plt.imshow(f.data, origin='lower', cmap=plt.cm.viridis, norm=LogNorm(),
           vmin=3E3, vmax=3E4)
plt.colorbar()

plt.xlabel('RA')
plt.ylabel('Dec')

plt.show()

################################################################################
# Centre the star positions
################################################################################

fig = plt.figure(figsize=(12, 12))
# fig.add_subplot(111)#, projection=my_wcs)

pic_order = [3, 4, 1, 2]

# M4/HAWKI.2015-06-10T05_12_28.683.fits
chip_distort_cens = [(1400, -300), (-300, 0), (1600, 1400), (-200, 1900)]
chip_stretchs = [(1.0175, 1.0175), (1.0175, 1.0175), (1.0175, 1.0175),
                 (1.0175, 1.0175)]
chip_offsets = [(0, 0), (0, 0), (0, 0), (0, 0)]

# chip_distort_cens = [(1600, 0), (0, 0), (0, 0), (0, 0)]
# chip_stretchs = [(1.01, 1), (1, 1), (1, 1), (1, 1)]
# chip_offsets  = [(0, 15), (-10, 18), (0, 13), (-10, 18)]
hdu_tbls = []

plt.figure(figsize=(10, 10))

for i in range(4)[:4]:
    hdu = fits.open(global_fname)[pic_order[i]]
    my_wcs = wcs.WCS(hdu.header)

    star_coords = SkyCoord(ra=oc_2mass["ra"], dec=oc_2mass["dec"], unit=u.deg)

    arr_coords_2mass = np.array((oc_2mass["ra"], oc_2mass["dec"])).T
    arr_pixels_2mass = my_wcs.wcs_world2pix(arr_coords_2mass, 1)

    mask_2mass = (arr_pixels_2mass[:, 0] > 20) * (
                arr_pixels_2mass[:, 0] < 2028) * \
                 (arr_pixels_2mass[:, 1] > 20) * (arr_pixels_2mass[:, 1] < 2028)

    ras = arr_pixels_2mass[:, 0][mask_2mass]
    decs = arr_pixels_2mass[:, 1][mask_2mass]

    # Apply the distorsion

    ra0 = chip_distort_cens[i][0]
    dec0 = chip_distort_cens[i][1]

    dras = ras - ra0
    ddecs = decs - dec0
    seps = np.sqrt(dras ** 2 + ddecs ** 2)
    angs = np.arctan2(decs, ras)

    ras_new = ra0 + dras * chip_stretchs[i][0] + chip_offsets[i][0]
    decs_new = dec0 + ddecs * chip_stretchs[i][1] + chip_offsets[i][1]

    ### Do the final centering
    im = hdu.data

    dp = 15
    n = 7
    for j in range(len(ras_new)):

        x = int(ras_new[j])
        y = int(decs_new[j])

        try:
            im2 = im[y - dp:y + dp + 1, x - dp:x + dp + 1]

            mean, median, std = sigma_clipped_stats(im2, sigma=3.0, iters=5)
            daofind = DAOStarFinder(fwhm=3.0, threshold=5. * std)
            sources = daofind(im2 - median)

            sharp_mask = sources["sharpness"] < 0.7
            flux_mask = np.argmax(sources["flux"][sharp_mask])

            xp, yp = sources["xcentroid"][sharp_mask][flux_mask], \
                     sources["ycentroid"][sharp_mask][flux_mask]
            dxp, dyp = int(xp - dp), int(yp - dp)

            ras_new[j] += dxp
            decs_new[j] += dyp

        except:
            pass

        if j % 100 == 0:
            print(j, len(ras_new))

    # convert the refined pixel coordinates back to RA, DEC
    arr_pixels_hawki = np.array((ras_new, decs_new)).T
    arr_coords_hawki = my_wcs.wcs_pix2world(arr_pixels_hawki, 1)

    oc_2mass_mini = oc_2mass[mask_2mass]

    hdu_tbls += [Table(data=[oc_2mass_mini["ra"], oc_2mass_mini["dec"],
                             ras, decs, ras_new, decs_new,
                             arr_coords_hawki[:, 0], arr_coords_hawki[:, 1],
                             oc_2mass_mini["j_m"], oc_2mass_mini["h_m"],
                             oc_2mass_mini["k_m"],
                             oc_2mass_mini["ph_qual"]],
                       names=["ra_2mass", "dec_2mass",
                              "x_2mass", "y_2mass", "x_corr", "y_corr",
                              "ra_corr", "dec_corr",
                              "j_m", "h_m", "k_m", "ph_qual"])]

    plt.subplot(2, 2, i + 1)
    plt.scatter(hdu_tbls[i]["x_corr"], hdu_tbls[i]["y_corr"], facecolors='none',
                edgecolors='r')
    # plt.scatter(hdu_tbls[0]["x_corr"], hdu_tbls[0]["y_corr"], facecolors='none', edgecolors='r')
    plt.imshow(hdu.data, origin='lower', cmap=plt.cm.viridis, norm=LogNorm(),
               vmin=5E3, vmax=3E4)

plt.show()

################################################################################
# Get the photometry for each star
################################################################################

plot = False
if plot:
    plt.figure(figsize=(12, 12))
plot_range = [0] if plot else [0, 1, 2, 3]

for i in plot_range:
    fluxes = []

    hdu = fits.open(global_fname)[pic_order[i]]
    # hdu = fits.open("M4/HAWKI.2015-06-10T05_12_28.683.fits")[pic_order[i]]
    im = hdu.data

    n = 7
    dp = 7

    tbl = hdu_tbls[i]

    q = n ** 2 if plot else len(tbl)
    for j in range(q):

        x = int(tbl["x_corr"][j])
        y = int(tbl["y_corr"][j])

        im2 = im[y - dp:y + dp + 1, x - dp:x + dp + 1]
        median = np.median(im2)
        im2 -= median

        im3 = im2[dp - 5:dp + 6, dp - 5:dp + 6]
        flux = np.sum(im3[im3 > 0])
        fluxes += [flux]

        if j % 100 == 0:
            print(j, len(tbl))

        if plot:
            plt.subplot(n, n, j + 1)
            plt.imshow(im2)

    if not plot:
        if "flux" in hdu_tbls[i].colnames:
            hdu_tbls[i].remove_column('flux')
        col = Column(data=np.nan_to_num(fluxes), name="flux")
        hdu_tbls[i].add_column(col)

if plot:
    plt.show()

################################################################################
# Check that we're getting meaningful values out for the
################################################################################

q=plt.hist(hdu_tbls[1]["flux"], bins=np.logspace(3,8,40))
plt.semilogx()
plt.show()

################################################################################
# Plot the raw HAWK-I magnitudes extracted from the chips to the 2MASS catalogue
################################################################################

# VLT and HAWKI data
area = 52
exptime = 10
gain = np.array([1.87, 1.735, 1.705, 2.11])
trans = 0.5
random_scale_factor = 0.5
F0 = simcado.source.zero_magnitude_photon_flux("Ks")  # Literature m=0 values for flux
H0 = area * exptime * trans / gain * random_scale_factor  # scale factor for go from ph/m2/s to ADU/dit for HAWKI

qual_dict = {"A": 1, "B": 0.6, 'C': 0.4, 'D': 0.2, 'E': 0, 'U': 0}

plt.figure(figsize=(10, 10))

for i, c in enumerate("rgbk"):
    plt.subplot(2, 2, i + 1)

    qual = [q[0] for q in hdu_tbls[i]["ph_qual"]]

    for z in "ABCDEU":
        mask = [q == z for q in qual]
        # mask = [True] * len(qual)

        # Literature flux based on 2MASS K-band mags in ph/m2/s scaled to ADU/dit for the VLT/HAWKI system
        flux_2mass = H0[i] * F0 * 10 ** (-0.4 * (hdu_tbls[i]["k_m"][mask]))
        # ADU/dit from the aperture photometry on the raw HAWKI images
        flux_hawki = hdu_tbls[i]["flux"][mask]

        plt.plot(flux_hawki, flux_2mass, c + "+", alpha=qual_dict[z])
        if z is "A": print(np.median(flux_2mass / flux_hawki))

    plt.plot([1E2, 1E8], [1E2, 1E8], "k--", alpha=0.5)

    plt.xlabel("Raw HAWK-I K-band counts")
    plt.ylabel("2MASS K-band flux")

    plt.xlim(1E4, 1E8);
    plt.ylim(1E4, 1E8)
    plt.loglog()

    plt.title("Chip " + str(pic_order[i]))
plt.tight_layout()
plt.show()

# zps = [26.7, 26.95, 26.95, 26.8]
# zps = [25.8]*4

for i in range(4):
    a = hdu_tbls[i]
    a.write("M4/M4_chip"+str(i)+'table.dat', format='ascii')
