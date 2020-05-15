The Plan
========

Photometry of raw HAWKI image
-----------------------------

* Import a fits file image,
* Load in the stars from astroquery
* Find stars with photutils
* Shift the image to match the photometry
* Get the photometry

Photometry of simulated HAWKI image
-----------------------------------

* Use the magnitudes from Simbad to generate a Simcado object
* Observe it with HAWKADO
* Find the stars with photutils
* Get the photometry

Comparison of the photometric values
------------------------------------

* Compare the two photometries
* Plot the two images next to each other
* Plot the photometries for J and Ks
* Generate the limiting magnitude vs time graph for HAWKI with the SNR contours