import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import csv

from synphot import SourceSpectrum, SpectralElement, units, etau_madau, Observation
from synphot.models import Empirical1D, ConstFlux1D
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
from astropy import units as u



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
BD_temp_path = data_array[179]
QSO_temp_path = data_array[171]
QSO_path = data_array[4]
Fil_path = data_array[1]

print("DATA INSIDE", BD_temp_path, "FILE")
# Using csv.reader to open csv files
with open(BD_temp_path,  'r') as file:
    BD_temp_reader = csv.reader(file, delimiter=' ')
    for row in BD_temp_reader:
        data_bd_temp = pd.DataFrame(BD_temp_reader)
print(data_bd_temp)

print("DATA INSIDE", QSO_temp_path, "FILE")
with open(QSO_temp_path, 'r') as file:
    QSO_temp_reader = csv.reader(file, delimiter=' ')
    for row in QSO_temp_reader:
        data_qso_temp = pd.DataFrame(QSO_temp_reader)
print(data_qso_temp)

print("DATA INSIDE", QSO_path, "FILE")
with open(QSO_path, 'r') as file:
    QSO_reader = csv.reader(file, delimiter=' ')
    for row in QSO_reader:
        data_qso = pd.DataFrame(QSO_reader)
print(data_qso)

print("DATA INSIDE", Fil_path, "FILE")
with open(Fil_path, 'r') as file:
    Fil_reader = csv.reader(file, delimiter=' ')
    for row in Fil_reader:
        data_fil = pd.DataFrame(Fil_reader, columns=["#id", "ra", "dec", "redshift", "g_prime_delve", "g_prime_err_delve", "r_prime_delve", "r_prime_err_delve", "i_prime_delve", "i_prime_err_delve", "z_prime_delve", "z_prime_err_delve", "vista.vircam.Y_vhs", "vista.vircam.Y_err_vhs", "vista.vircam.J_vhs", "vista.vircam.J_err_vhs", "vista.vircam.H_vhs", "vista.vircam.H_err_vhs", "vista.vircam.Ks_vhs", "vista.vircam.Ks_err_vhs", "WISE1", "WISE1_err", "WISE2", "WISE2_err"])
print(data_fil)

# III: Make column-vectors from BD templated
BD_g_decam = data_bd_temp[1].astype(float)
BD_r_decam = data_bd_temp[2].astype(float)
BD_i_decam = data_bd_temp[3].astype(float)
BD_z_decam = data_bd_temp[4].astype(float)
BD_Y_vhs = data_bd_temp[5].astype(float)
BD_J_vhs = data_bd_temp[6].astype(float)
BD_H_vhs = data_bd_temp[7].astype(float)
BD_K_vhs = data_bd_temp[8].astype(float)
BD_W1 = data_bd_temp[9].astype(float)
BD_W2 = data_bd_temp[10].astype(float)
BD1_vec_flux = np.array([BD_g_decam[0], BD_r_decam[0], BD_i_decam[0], BD_z_decam[0], BD_Y_vhs[0], BD_J_vhs[0], BD_H_vhs[0], BD_K_vhs[0], BD_W1[0], BD_W2[0]])
BD2_vec_flux = np.array([BD_g_decam[1], BD_r_decam[1], BD_i_decam[1], BD_z_decam[1], BD_Y_vhs[1], BD_J_vhs[1], BD_H_vhs[1], BD_K_vhs[1], BD_W1[1], BD_W2[1]])
BD3_vec_flux = np.array([BD_g_decam[2], BD_r_decam[2], BD_i_decam[2], BD_z_decam[2], BD_Y_vhs[2], BD_J_vhs[2], BD_H_vhs[2], BD_K_vhs[2], BD_W1[2], BD_W2[2]])
BD4_vec_flux = np.array([BD_g_decam[3], BD_r_decam[3], BD_i_decam[3], BD_z_decam[3], BD_Y_vhs[3], BD_J_vhs[3], BD_H_vhs[3], BD_K_vhs[3], BD_W1[3], BD_W2[3]])
BD_all_vec_flux = np.array([BD1_vec_flux, BD2_vec_flux, BD3_vec_flux, BD4_vec_flux])



# ---Read QSO one template for the example
dQSO_raw = data_qso_temp # ---redshift z=6
list_qso = []
for i in range(len(dQSO_raw)):
    list_qso.append([dQSO_raw.iloc[i, 1], dQSO_raw.iloc[i, 2], dQSO_raw.iloc[i, 3], dQSO_raw.iloc[i, 4], dQSO_raw.iloc[i, 5], dQSO_raw.iloc[i, 6], dQSO_raw.iloc[i, 7], dQSO_raw.iloc[i, 8], dQSO_raw.iloc[i, 9], dQSO_raw.iloc[i, 10]])
dQSO_vec_flux = np.array(list_qso)
dQSO_vec_flux = dQSO_vec_flux.astype(float)
print("QSO filters ", dQSO_vec_flux)
# qso_S15_wave = data_qso[data_qso.columns[0]]
# qso_S15_wave = qso_S15_wave.astype(float)
# qso_S15_flux = data_qso[data_qso.columns[1]]
# qso_S15_flux = qso_S15_wave.astype(float)
# print("Flux of the source, qso_S15_flux)


# ---- Define functions for calculating scaling factor and Chi2 for convenience
def a_scale(vec_flux_obs, vec_fluxe_obs, vec_flux_model):
    # ---- Obtain scaling factor for Chi2
    a = np.sum((vec_flux_obs * vec_flux_model) / (vec_fluxe_obs) ** 2) / np.sum(
        (vec_flux_model) ** 2 / (vec_fluxe_obs) ** 2)
    return a


def chi2_calc(vec_flux_obs, vec_fluxe_obs, vec_flux_model, a):
    # ---- Obtain scaling factor for Chi2
    chi2 = np.sum((vec_flux_obs - a * vec_flux_model) ** 2 / (vec_fluxe_obs) ** 2)
    return chi2


# # -----Define central wavelength of source for plot it later for check
# Solution of reading the file automatically and making array with filters
# ---- It is better to create a numpy array from the observed fluxes, observed fluxes errors, and template fluxes
#     this is because the code will be much faster dealing with np.array than doing for loops.
# data = fits.getdata("stripe82_totalmag_1007_truncated.fits")

header = list(data_fil.columns)
select_err = "err"
del header[:4]
errors = [i for i in header if select_err in i]
wavebands = [i for i in header if i not in errors]

# dataset_header = [elem.replace(ch, '') for elem in dataset_header]
# # making final array of the list
print("WAVEBANDS")
print(wavebands)

# # Make array for central wavelengths of the bands(each of them) /// Hard to find wavelengths in Surveys web pages (need to be checked again)
flux_array = data_fil[["g_prime_delve", "r_prime_delve", "i_prime_delve", "z_prime_delve", "vista.vircam.Y_vhs", "vista.vircam.J_vhs", "vista.vircam.H_vhs", "vista.vircam.Ks_vhs", "WISE1", "WISE2"]].copy()
flux_array.columns = [''] * len(flux_array.columns)
fil_values = np.array(flux_array)
vec_flux_obs = flux_array.astype(float)



# # Make another array for width of wavelengths of the bands(each of them)
print("WAVEBANDS ERRORS")
print(errors)
error_array = data_fil[["g_prime_err_delve", "r_prime_err_delve", "i_prime_err_delve", "z_prime_err_delve", "vista.vircam.Y_err_vhs", "vista.vircam.J_err_vhs", "vista.vircam.H_err_vhs", "vista.vircam.Ks_err_vhs", "WISE1_err", "WISE2_err"]].copy()
error_array.columns = [''] * len(error_array.columns)
err_values = np.array(error_array)
vec_fluxe_obs = err_values.astype(float)


vec_flux_model_BD = BD_all_vec_flux.astype(float)
vec_flux_model_QSO = dQSO_vec_flux.astype(float)


# ---Calculate scaling factor and Chi2 for BDs
a_BD = a_scale([vec_flux_obs, vec_fluxe_obs, vec_flux_model_BD[0]])
a_BD = a_BD.astype(float)
Chi2_BD = chi2_calc(vec_flux_obs, vec_fluxe_obs, vec_flux_model_BD[0], a_BD)

# ---Calculate scaling factor and Chi2 for QSOs
a_QSO = a_scale(vec_flux_obs, vec_fluxe_obs, vec_flux_model_QSO[0])
a_QSO = a_QSO.astype(float)
Chi2_QSO = chi2_calc(vec_flux_obs, vec_fluxe_obs, vec_flux_model_QSO[0], a_QSO)

print("===============================================")
print("Chi2 BD:", Chi2_BD)
print("Chi2 QSO:", Chi2_QSO)

# # ----Plots
# plt.plot(ll_vec_red, vec_flux_model_BD * a_BD, 'o', color='k')
# plt.plot(ll_vec_red, vec_flux_model_QSO * a_QSO, 'o', color='b')
# plt.errorbar(ll_vec_red, vec_flux_obs, yerr=vec_fluxe_obs, fmt='o', color='r')
# # plt.plot(bdRA_l, bdRA_f1 * a_BD, color='gray')
# plt.plot(qso_S15_wave * (1 + dQSO_red), qso_S15_flux * a_QSO, color='b')
# plt.savefig("plot.png")
