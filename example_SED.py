import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import csv

from synphot import SourceSpectrum, SpectralElement, units, etau_madau, Observation
from synphot.models import Empirical1D, ConstFlux1D
from astropy.io import ascii
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
from astropy import units

def flux_from_magAB(magAB,ll):
    #---- Obtain flux from AB magnitude and wavelenght (in Angstrom) of the band
    flux = 10**((23.9-magAB)/2.5) * 10**(-11)/ll**2
    return flux

#---- path where SED code lives for now
#---- to be changed in the according path in your laptop
path = '/home/dathev/PhD_Projects/4MOST/chicode_sm/'

#--- some plotting parameters
plt.rcParams['figure.figsize'] = [20, 12]  #--- resize plots
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20

#--- path where filter curves are.
#---- to be changed in the according path in your laptop
path_filters = "/home/dathev/PhD_Projects/4MOST/chicode_sm/mag_from_spec/filters/DECam/"

#--- Read the filters files.
#--- !! for now I am reading the decam filters but we will need to change to the DELVE etc filters!!
# g_name = get_pkg_data_filename(os.path.join(path_filters, 'filter_g_decam.tab'))
g_name = g_name = path_filters + 'filter_g_decam.tab'
bp_g = SpectralElement.from_file(g_name)
# r_name = get_pkg_data_filename(os.path.join(path_filters, 'filter_r_decam.tab'))
r_name = r_name = path_filters + 'filter_r_decam.tab'
bp_r = SpectralElement.from_file(r_name)
# z_name = get_pkg_data_filename(os.path.join(path_filters, 'filter_z_decam.tab'))
z_name = z_name = path_filters + 'filter_z_decam.tab'
bp_z = SpectralElement.from_file(z_name)

# Plotting the filters
plt.plot(bp_g.waveset, bp_g(bp_g.waveset), 'b', label='g decam')
plt.plot(bp_r.waveset, bp_r(bp_r.waveset), 'g', label='r decam')
plt.plot(bp_z.waveset, bp_z(bp_z.waveset), 'r', label='z decam')
plt.legend(loc='upper right', fontsize=22)
plt.xlabel('Wavelength (Angstrom)', fontsize=20)
plt.ylabel('Transmission', fontsize=20)
plt.title("Filter Set", fontsize=20)
plt.savefig("example_SED_filters.png")


#----Calculate the effective wavelength of the filters by interpolating with a reference spectrum flat in AB system
wave_ref = range(900, 26000, 10)
sp_ref = SourceSpectrum(ConstFlux1D, amplitude=18 * units.ABmag)

obs_ref_g = Observation(sp_ref, bp_g, binset=bp_g.waveset, force='extrap')
lleff_ref_g = obs_ref_g.effective_wavelength
print("Effective wavelength g:", lleff_ref_g())

obs_ref_r = Observation(sp_ref, bp_r, binset=bp_r.waveset, force='extrap')
lleff_ref_r = obs_ref_r.effective_wavelength
print("Effective wavelength r:", lleff_ref_r())

obs_ref_z = Observation(sp_ref, bp_z, binset=bp_z.waveset, force='extrap')
lleff_ref_z = obs_ref_z.effective_wavelength
print("Effective wavelength z:", lleff_ref_z())

#----Plotting Brown Dwarf Spectrum (available in read.py)