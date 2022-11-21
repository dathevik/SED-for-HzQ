# This is a test version of 4MOST project

import numpy as np
import pandas as pd
import os
import csv
from synphot import SourceSpectrum, units, SpectralElement, etau_madau, Observation
from synphot.models import Empirical1D, BlackBodyNorm1D, ConstFlux1D
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy import units as u
from astropy.io import ascii
from astropy.utils.data import get_pkg_data_filename
from scipy.stats import chisquare

# Find data files from all folders and subfolders
# Get the list of all files in a directory and read them all
# Mak2e the data columns independent vectors
# Read fluxes, plot it and calculate effective wavelength
# Input magnitudes, plot the spectrum, normalized and redshifted
# Obtain fluxes from the spectrum

# --- some plotting parameters
plt.rcParams['figure.figsize'] = [20, 12]  # --- resize plots
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20

# I: Define the path and print a list of files from folders and subfolders
my_path = r'/home/dathev/PhD_Projects/4MOST/chicode_sm'
new_array = []  # make empty 1D array to collect files in the array
count_the_files = -1
print("THIS IS THE LIST OF ALL FILES -------------")
# find all data files using loop
for root, subFolder, files in os.walk(my_path):
    for fileName in files:
        if (fileName.endswith("txt")
                or fileName.endswith("dat")
                or fileName.endswith("tab")
                or fileName.endswith("spc")):  # find files of the mentioned formats
            filePath = os.path.join(root, fileName)
            count_the_files += 1
            new_array.append(filePath)  # make a list of files
            print(count_the_files, filePath)

# II: Use Pandas to make data of the files into a Dataframe
print(end="\n\n\n")
data_array = np.array(new_array)
read_file = data_array[175]
print("DATA INSIDE", read_file, "FILE")
# Using csv.reader to open csv files
with open(read_file, 'r') as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        data = pd.DataFrame(reader)
print(data)

# III: Make column-vector and write output in other file also skipping some rows
data = data.drop(data.index[:14])
wave = data[data.columns[0]]
flux = data[data.columns[1]]
error = data[data.columns[2]]
wave = wave.astype(float) * 10000  # ---in Angstrom
flux = flux.astype(float) * units.FLAM
error = error.astype(float)
wave = np.array(wave.values)
flux = np.array(flux.values)
error = np.array(error.values)
print("Flux column from ", read_file, "is ", flux)

# open new txt file and write there the column
# new_f = open('/home/dathev/PhD_Projects/4MOST/chicode_sm/data_array.txt', "w")
# new_f.write(str(column_vector))

# IV: Define path of filters, read filters and plot them
path_filters = r"/home/dathev/PhD_Projects/4MOST/chicode_sm/mag_from_spec/filters/DECam/"


def flux_from_magAB(magAB, ll):
    # ---- Obtain flux from AB magnitude and wavelenght (in Angstrom) of the band
    flux = 10 ** ((23.9 - magAB) / 2.5) * 10 ** (-11) / ll ** 2
    return flux


# --- Read the filters files.
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
plt.rcParams['figure.figsize'] = [20, 12]
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.plot(bp_g.waveset, bp_g(bp_g.waveset), 'b', label='g decam')
plt.plot(bp_r.waveset, bp_r(bp_r.waveset), 'g', label='r decam')
plt.plot(bp_z.waveset, bp_z(bp_z.waveset), 'r', label='z decam')
plt.legend(loc='upper right', fontsize=22)
plt.xlabel('Wavelength (Angstrom)', fontsize=20)
plt.ylabel('Transmission', fontsize=20)
plt.title("Filter Set", fontsize=20)
plt.savefig(my_path + "/output/SED_filters.png")
plt.close()

# ----Calculate the effective wavelength of the filters by interpolating with a reference spectrum flat in AB system
wave_ref = range(900, 26000, 10)
sp_ref = SourceSpectrum(ConstFlux1D, amplitude=18 * u.ABmag)

obs_ref_g = Observation(sp_ref, bp_g, binset=bp_g.waveset, force='extrap')
lleff_ref_g = obs_ref_g.effective_wavelength
print("Effective wavelength g:", lleff_ref_g())

obs_ref_r = Observation(sp_ref, bp_r, binset=bp_r.waveset, force='extrap')
lleff_ref_r = obs_ref_r.effective_wavelength
print("Effective wavelength r:", lleff_ref_r())

obs_ref_z = Observation(sp_ref, bp_z, binset=bp_z.waveset, force='extrap')
lleff_ref_z = obs_ref_z.effective_wavelength
print("Effective wavelength z:", lleff_ref_z())

# ---- This is eventually read from a table
# ---- Magnitudes are in AB
mag_g = 23.74;
mag_g_e = 0.33
mag_r = 23.61;
mag_r_e = 0.33
mag_z = 19.64;
mag_i_e = 0.02

# ---calculate fluxes from observed magnitudes
flux_g = flux_from_magAB(mag_g, lleff_ref_g().value)
flux_r = flux_from_magAB(mag_r, lleff_ref_r().value)
flux_z = flux_from_magAB(mag_z, lleff_ref_z().value)
# ---TBD: calculate errors on fluxes


# # V: Make a spectrum from the file selected rows
sp = SourceSpectrum(Empirical1D, points=wave, lookup_table=flux, keep_neg=True)
sp.plot()
plt.rcParams['figure.figsize'] = [20, 12]
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.savefig(my_path + "/output/spectrum.png")
plt.close()

# ---- Normalize the spectrum at a given flux, taken from the observed object (test in our case here)

sp_norm = sp.normalize(flux_z, band=bp_z)
wave_norm = sp_norm.waveset
plt.rcParams['figure.figsize'] = [20, 12]  # --- resize plots
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.plot(wave_norm, sp_norm(wave_norm), 'k', label='L Dwarf')
plt.plot(lleff_ref_z().value, flux_z, 'o', color='r', markersize=15, label="Flux z band")
plt.legend(loc='upper right', fontsize=22)
plt.xlabel('Wavelength (Angstrom)', fontsize=20)
plt.ylabel('Flux (erg/s/cm2/A) ', fontsize=20)
plt.title("Brown Dwarf (LDwarf) Normalized at z band", fontsize=20)
plt.savefig(my_path + "/output/normalized_spectrum.png")
plt.close()

# Changing redshift of the spectrum
sp_z1 = SourceSpectrum(sp.model, z=1, z_type='conserve_flux')
sp_z3 = SourceSpectrum(sp.model, z=3, z_type='conserve_flux')
sp_z5 = SourceSpectrum(sp.model, z=5, z_type='conserve_flux')
sp_z6 = SourceSpectrum(sp.model, z=6, z_type='conserve_flux')
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.plot(wave, sp(wave), 'grey', label='rest frame')
plt.plot(wave, sp_z1(wave), 'b', label='z=1')
plt.plot(wave, sp_z3(wave), 'g', label='z=3')
plt.plot(wave, sp_z5(wave), 'orange', label='z=5')
plt.plot(wave, sp_z6(wave), 'r', label='z=6')
plt.legend(loc='upper right', fontsize=22)
plt.xlabel('Wavelength (Angstrom)', fontsize=20)
plt.ylabel('Flux (not normalized)', fontsize=20)
plt.savefig(my_path + "/output/redshifted_spectrum.png")
plt.close()

# ---Apply the absorption from the Intergalactic medium
extcurve_1 = etau_madau(wave, 1)
sp_ext1 = sp * extcurve_1
extcurve_3 = etau_madau(wave, 3)
sp_ext3 = sp * extcurve_3
extcurve_5 = etau_madau(wave, 5)
sp_ext5 = sp * extcurve_5
extcurve_6 = etau_madau(wave, 6)
sp_ext6 = sp * extcurve_6

plt.plot(wave, sp(wave), 'grey', label='rest frame')
plt.plot(wave, sp_ext1(wave), 'b', label='z=1')
plt.plot(wave, sp_ext3(wave), 'g', label='z=3')
plt.plot(wave, sp_ext5(wave), 'orange', label='z=5')
plt.plot(wave, sp_ext6(wave), 'r', label='z=6')
plt.legend(loc='upper right', fontsize=22)
plt.xlabel('Wavelength (Angstrom)', fontsize=20)
plt.ylabel('Flux (not normalized)', fontsize=20)
plt.title("IGM Absoprtion", fontsize=20)
plt.rcParams['figure.figsize'] = [20, 12]
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.savefig(my_path + '/output/IGM.png')
plt.close()

# ---- Obtain fluxes from the spectra convolving with the passbands
obs_g = Observation(sp_norm, bp_g, binset=bp_g.waveset, force='extrap')
obs_r = Observation(sp_norm, bp_r, binset=bp_r.waveset, force='extrap')
obs_z = Observation(sp_norm, bp_z, binset=bp_z.waveset, force='extrap')

# ----calculate fluxes
sp_flux_g = obs_g.effstim()
sp_flux_r = obs_r.effstim()
sp_flux_z = obs_z.effstim()

# -check our normalization, with plotting the part of the spectrum covered by one band alone
sp_z = obs_z.as_spectrum(obs_z)
sp_z.plot()
plt.plot(lleff_ref_z().value, sp_flux_z, 'o', color='g', markersize=15, label="Flux z band effective")
plt.plot(lleff_ref_z().value, flux_z, 'o', color='b', markersize=15, label="Flux z observed")
plt.savefig(my_path + "/output/overlap.png")
plt.close()

# ---Let's plot now the spectrum with the derived synthetic fluxes, and the observed fluxes
plt.plot(wave_norm, sp_norm(wave_norm), 'k', label='L Dwarf')

# ---observed fluxes
plt.plot(lleff_ref_r().value, flux_r, 'o', color='g', markersize=15, label="Syn Flux r band")
plt.plot(lleff_ref_g().value, flux_g, 'o', color='orange', markersize=15, label="Syn Flux r band")
plt.plot(lleff_ref_z().value, flux_z, 'o', color='r', markersize=15, label="Syn Flux z band")

# ---synthetic fluxes
plt.plot(lleff_ref_r().value, sp_flux_r,
         markeredgecolor='g', markerfacecolor='None', markeredgewidth=3,
         marker='s', markersize=15, label="Obs Flux r band")
plt.plot(lleff_ref_g().value, sp_flux_g,
         markeredgecolor='orange', markerfacecolor='None', markeredgewidth=3,
         marker='s', markersize=15, label="Obs Flux g band")
plt.plot(lleff_ref_z().value, sp_flux_z,
         markeredgecolor='r', markerfacecolor='None', markeredgewidth=3,
         marker='s', markersize=15, label="Obs Flux z band")

plt.legend(loc='upper right', fontsize=22)
plt.xlabel('Wavelength (Angstrom)', fontsize=20)
plt.ylabel('Flux (erg/s/cm2/A) ', fontsize=20)
plt.title("Brown Dwarf (LDwarf) Normalized at z band and fluxes", fontsize=20)
plt.savefig(my_path + '/output/normalized_with_filters.png')
plt.close()


