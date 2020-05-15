from astropy.io import ascii
from astropy import units as u

import numpy as np
import matplotlib.pyplot as plt


def mu_to_d(mu):
    return 10**((mu+5)/5)


def d_to_mu(d):
    return 5*np.log10(d)-5


# K band absolute mags taken from here:
# http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt

wars = ascii.read("spec_tiny.dat")
print(wars["Mass"][0]*1000)


filt = "M_J"

#Exposure times for 5 sigma point source with MCAO
exptime = ["   2.6 sec Saturation Limit", "1 hour (VLT/HAWKI)",
           "2.6 sec (5$\sigma$, SCAO)", "1 min", "1 hour", "5 hour"]

if "J" in filt:
    K_exp_mag = [15.9, 23.9, 23.4, 25.6, 27.8, 28.7]
    label_offset = [0,0.5,0,0,0,0]
elif "K" in filt:
    K_exp_mag = [14.8, 22.3, 22.4, 24.1, 26.4, 27.3]
    label_offset = [0,0,0.5,0,0,0]

#Distance of well known objects in kpc
obj = ["130pc (Pleiades)", "450pc (Orion)", "8.5kpc (No extinction)", "50kpc (LMC)",
       "770kpc (M31)", "2Mpc (NGC300)", "4Mpc (Cen A)", "5Mpc (M83)", "20Mpc (Virgo)"]
dist = [0.13,      0.45,    8.5,                  50,   770,   2150,     3815,    4610,  20000]

#Colours for spectral types
clr = ["#B2EEEC","#B2EEEC","#DAE0E0","#DAE0E0","#E9ECB8","#E4EE40","#EECB40","#EDB279","#ED8C79","#DD7C69","#C3610F","#C3610F"]
#        O9I        O6          B0         B3      A0         F0        G0       K0         M0        L0        T0        T8

#Find the distances for distance moduli of 0 to 37
mu = np.arange(37).astype(float)
d = mu_to_d(mu).astype(float)


plt.figure(figsize=(20, 12))

#Stars - apparent magnitude of each spectral type vs distance
for i in range(len(wars[filt])):
    plt.plot(d/1E3, mu+wars[filt][i], c=clr[i], linewidth=8)
for i in range(len(wars[filt])):
    plt.text(mu_to_d(2.9-wars[filt][i]),18, wars["SpT"][i], fontsize=18, rotation=-51)
for i in range(8,len(wars[filt])):
    msg = "("+str(int(wars["Mass"][i]*1000))+"$M_{Jup})$"
    plt.text(mu_to_d(3.8-wars[filt][i]),19, msg, fontsize=18, rotation=-45)

#Well known objects - vertical dashed line and dot point
text_dist = np.array([0.8]*(len(obj)-2)+[1.12,0.80])
for j in range(len(dist)): plt.plot((dist[j], dist[j]),(0,32),"k--", alpha=0.5)
for j in range(len(obj)):  plt.text(text_dist[j]*dist[j], 29.5, obj[j], verticalalignment="bottom",
                                    horizontalalignment="left", fontsize=18, rotation=90)
#for j in range(len(obj)):  plt.scatter(dist[j], 28.3-1.2*j)

# Exposure times - horizontal dotted line for sec, min, hour
for j in range(len(K_exp_mag)): plt.plot((0.001,100000), (K_exp_mag[j], K_exp_mag[j]), "k:")
for j in range(len(exptime)): plt.text(1.2E-2, K_exp_mag[j]-0.1+label_offset[j], exptime[j], fontsize=18)

offset = 31.8
for i in np.arange(-2,5): plt.text(10.**i, offset, str(5*(i+2))+" mag", horizontalalignment="center", fontsize=18)
plt.text(20, offset+0.7,"Distance Modulus", horizontalalignment="center", fontsize=18)


plt.semilogx()
plt.xlabel("Distance [kpc]",fontsize=18)
plt.ylabel("Apparent "+filt[2:]+"-band magnitude [mag]", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.xlim(0.01, 30000)
plt.gca().yaxis.set_ticks_position('left')
#plt.ylim(14,28)

ax2 = plt.gca().twiny()

plt.semilogx()
ax2.set_xlim(0.01,30000)
ax2.set_ylim(14,30)

ax2.set_xticklabels([int(i) if i>1 else float(i) for i in ax2.get_xticks()*4])
ax2.set_xlabel("On sky distance per wide-field (4mas) pixel [AU]", fontsize=18)
ax2.tick_params(axis='both', which='major', labelsize=18)

plt.gca().invert_yaxis()

plt.savefig("spec_type_vs_dist_"+filt[2:]+".pdf",transparent=False, format="pdf")
