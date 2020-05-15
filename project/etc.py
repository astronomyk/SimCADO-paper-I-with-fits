import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import simcado


class ETC():
    """
    Basic class for exposure time calculator functionality

    Parameters
    ----------
    cmds : simcado.UserCommands
        Contains the configuration of the optical system

    opt_train : simcado.OpticalTrain, optional
        If provided, the ETC will not initialise a new ``OpticalTrain`` based on
        the ``UserCommands`` object passed in ``cmds``

    inst_filter : str, simcado.TransmissionCurve, optional
        If provided, ``inst_filter`` will override the filter curve given by ``cmds``

    star_filter : str, optional
        The magnitude of the test stars will use this filter, rather than the filter
        using to observe the stars. Any in-built simcado filter name is allowed
        See ``simcado.optics.get_filter_set()``

    mmin, mmax : float, optional
        [mag] The minimum and maximum magnitudes for the test stars. 100 stars in this range are
        observed


    See also
    --------
    ``simcado.optics.get_filter_set()``


    """

    def __init__(self, cmds, opt_train=None, inst_filter=None, star_filter=None,
                 mmin=18, mmax=28, n_stars=100, grid_spacing=100,
                 snr_fitting_limits=[10, 200]):

        self.image = None
        self.hdu = None
        self.cmds = cmds
        self.snr_fitting_limits = snr_fitting_limits

        self.mag_min = np.min((mmin, mmax))
        self.mag_max = np.max((mmin, mmax))
        # self.spec_type = spec_type
        self.n_stars = n_stars
        self.pixel_sep = grid_spacing

        self.star_filter = self.cmds[
            "INST_FILTER_TC"] if star_filter is None else star_filter
        self.star_sep = self.pixel_sep * self.cmds["SIM_DETECTOR_PIX_SCALE"]
        self.star_grid = simcado.source.star_grid(n=self.n_stars,
                                                  mag_min=self.mag_min,
                                                  mag_max=self.mag_max,
                                                  filter_name=self.star_filter,
                                                  separation=self.star_sep)

        self.mags = np.linspace(self.mag_min, self.mag_max, self.n_stars)

        if inst_filter is not None: self.cmds["INST_FILTER_TC"] = inst_filter
        self.inst_filter = self.cmds["INST_FILTER_TC"]

        self.optical_train = simcado.OpticalTrain(
            cmds=self.cmds) if opt_train is None else opt_train

        cmds["OBS_NDIT"] = 1
        cmds["FPA_LINEARITY_CURVE"] = "none"
        cmds["FPA_CHIP_LAYOUT"] = "small"

        self.detector = simcado.Detector(cmds=self.cmds)

        self.star_grid.apply_optical_train(self.optical_train, self.detector)
        self.x = self.star_grid.x_pix
        self.y = self.star_grid.y_pix

    def extract_star(self, x, y, w):
        """
        Extract a sub-images of the test field

        Parameter
        ---------
        x, y : int, list
            [pixel] Central coordinate(s) of the sub-image to be extracted
        w : int
            [pixel] Width of sub-image(s)


        Returns
        -------
        ims : list
            list of numpy 2D arrays

        """

        if self.image is not None:

            x = np.array([x], dtype=int) if np.isscalar(x) else x.astype(int)
            y = np.array([y], dtype=int) if np.isscalar(y) else y.astype(int)

            ims = []
            for xi, yi in zip(x, y):

                try:
                    ims += [self.image[yi - w:yi + w, xi - w:xi + w]]
                except:
                    ims += [None]

            return ims

        else:
            raise ValueError("No image has been created")

    def snr(self, exptimes=3600, mags=None, aperture_width=10, bg_width=5,
            bg_in=None, fitted=False):
        """
        Return the signal-to-noise ratio for all stars in the test field based on aperture photometry

        Parameters
        ----------
        exptimes : float, list, optional
            [s] Exposure times to observe the test field. Default is 1 hr (3600s)

        mags : float, int, optional
            [mag] If only the SNR for specific magnitudes is required. If None, SNRs for all stars in
            the test field are returned

        aperture_width : int, optional
            [pixel] The width of the aperture placed around each star for photometry

        bg_width : int, optional
            [pixel] The width of the annulus placed around each star to get the background flux

        bg_in : int, optional
            [pixel] The inner radius of the background aperture. If ´´None´´ then ´´bg_in = aperture width´´.

        fitted : bool
            Default False. If True, a curve is fitted to the list of SNRs for each exposure time
            in order to give meaning full SNRs for faint stars with unreliable photometry


        Returns
        -------
        snrs : float, list
            A list, or list of lists, with signal-to-noise ratios for all stars in the test field
            and for all exposure times given. Corresponding magnitudes are in ``<ETC>.mags``

        """

        if bg_in is None: bg_in = aperture_width

        if not np.isscalar(exptimes):
            snrs = [
                self.snr(exptime, mags, aperture_width, bg_width, bg_in, fitted)
                for exptime in exptimes]

        else:

            from scipy import stats

            self.hdu = self.detector.read_out(OBS_EXPTIME=exptimes)
            self.image = self.hdu[0].data

            ap_out = aperture_width
            bg_out = bg_in + bg_width
            dw = bg_out - ap_out

            ims = self.extract_star(self.x, self.y, bg_out)

            signals, noises, snrs, bgs = [], [], [], []
            for im in ims:

                if im is not None:
                    sig = np.copy(im[dw:-dw, dw:-dw])
                    bg = np.ma.array(np.copy(im))
                    bg[bg_width:-bg_width, bg_width:-bg_width] = 0

                    bgs += [np.median(bg[bg != 0])]
                    noises += [np.std(bg[bg != 0]) * np.sqrt(np.sum(bg == 0))]
                    signals += [np.sum(sig - bgs[-1])]

            noises = np.array(noises)
            signals = np.array(signals)
            snrs = signals / noises

            if fitted:
                mask = (snrs > self.snr_fitting_limits[0]) * \
                       (snrs < self.snr_fitting_limits[1])
                q = stats.linregress(self.mags[mask], np.log10(snrs[mask]))
                snrs = 10 ** (q.intercept + q.slope * self.mags)

            if mags is not None:
                user_snrs = []
                if np.isscalar(mags): mags = [mags]

                for mag in mags:
                    i = np.where(self.mags > mag)[0][0]
                    if i is None or i < 1:
                        raise ValueError("m=" + str(
                            mag) + " outside range tested. Remake ETC with lower mmin")
                    else:
                        user_snrs += [
                            np.interp([mag], self.mags[i - 1:i], snrs[i - 1:i])]
                snrs = user_snrs

        return snrs

    def limiting_magnitudes(self, exptimes=3600, limiting_sigmas=[5],
                            plot=False, **kwargs):
        """
        Get the limiting magnitude for a set exposure times and set of noise levels

        Parameters
        ----------
        exptimes : float, list, optional
            [s] Default 1 hr (3600s). A single or list of exposure times

        limiting_sigmas : float, list, optional
            Default 5. The noise level(s) which define a detection (e.g. 5 sigma, 10 sigma, etc)

        plot : bool, optional
            Default False. If True, the SNR for a stars in the test field is plotted against exposure time

        **kwargs are passed directly to <ETC>.snr()


        Returns
        -------
        limiting_mags : float, list
            [mag] A single or list of limiting magnitudes for each exposure time and detection limit level


        See also
        --------
        <ETC>.snr()

        """

        if not np.isscalar(exptimes):
            limiting_mags = [
                self.limiting_magnitudes(exptime, limiting_sigmas, plot,
                                         **kwargs) for exptime in exptimes]

        else:
            from scipy import stats

            if np.isscalar(limiting_sigmas): limiting_sigmas = [limiting_sigmas]

            snrs = self.snr(exptimes=exptimes, **kwargs)
            mags = self.mags

            mask = (snrs > self.snr_fitting_limits[0]) * \
                   (snrs < self.snr_fitting_limits[1])
            q = stats.linregress(mags[mask], np.log10(snrs[mask]))
            limiting_mags = (np.log10(limiting_sigmas) - q.intercept) / q.slope

            if plot:
                x = np.linspace(self.mag_min, self.mag_max, 100)
                plt.plot(x, 10 ** (q.slope * x + q.intercept), "k--",
                         label=str(exptimes) + "s")
                plt.plot(self.mags, snrs, "b+")
                for m, s in zip(limiting_mags, limiting_sigmas):
                    plt.axvline(m, ls=":", c="g")
                    plt.axhline(s, ls=":", c="r")

                plt.semilogy()
                plt.xlabel(self.inst_filter + " [mag]")
                plt.ylabel("S/N")

        return limiting_mags

    def plot_snr_rainbow(self, exptimes, snr_array, snr_levels=[1, 5, 10, 250],
                         text_heights=None, text_center=1, use_colorbar=True):
        """
        Plot a nice rainbow curve of the SNR as a function of exposure time and magnitude

        magnitudes are taken from the internal <ETC>.mags attribute

        Parameters
        ----------
        exptimes : list, np.ndarray
            Exposure times, in whatever unit you want them to be displayed

        snr_array : 2D np.ndarray
            A 2D (n,m) array where n is the length of ''exptimes'' and m is the length of ''mags''

        snr_levels : list, np.ndarray
            Which contours should be plotted on the graph. Default is [1, 5, 10, 250] sigma.

        text_height : list, np.ndarray
           [mag] The height at which the contour labels should be plotted. Default is ''None''

        text_center : float
            The position along the x (time) axis where the detection limit label should be placed


        Returns
        -------
        fig : matplotlib.Figure object


        """

        # fig = plt.figure(figsize=(10,5))
        plt.contour(exptimes, self.mags, np.array(snr_array).T, snr_levels,
                    colors=list("krygbkkkkkkkk"))

        lvls = list(range(1, 10)) + list(range(10, 100, 10)) + list(
            range(100, 1001, 100))
        plt.contourf(exptimes, self.mags, np.array(snr_array).T, levels=lvls,
                     norm=LogNorm(),
                     alpha=0.5, cmap="rainbow_r")
        if use_colorbar:
            clb = plt.colorbar()
            clb.set_label("Signal to Noise Ratio ($\sigma$)")

        if text_heights is not None:
            for height, s in zip(text_heights, snr_levels):
                plt.text(text_center, height, str(s) + "$\sigma$", rotation=0)

        plt.grid()

        plt.xlim(exptimes[0], exptimes[-1])
        plt.ylim(self.mags[0], self.mags[-1])
