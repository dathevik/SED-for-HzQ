import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii

# I: Define the path and print the data inside
filters_path = r'/home/dathev/PhD_Projects/4MOST/chicode_sm/cigale_flux_ALL-SKY_delve.dat'
BD_temp_path = r'/home/dathev/PhD_Projects/4MOST/chicode_sm/BDspectra/BDRA_fluxes_mJy.dat'
QSO_temp_path = r'/home/dathev/PhD_Projects/4MOST/chicode_sm/temp_qso/Selsing2015_fluxes_mJy.dat'

print("DATA INSIDE", BD_temp_path, "FILE")
data_bd_temp = ascii.read(BD_temp_path)
print(data_bd_temp)

print("DATA INSIDE", QSO_temp_path, "FILE")
data_qso_temp = ascii.read(QSO_temp_path)
print(data_qso_temp)

print("DATA INSIDE", filters_path, "FILE")
data_fil = ascii.read(filters_path)
print(data_fil)

# IIa : Make vector arrays for Brown Dwarf templates
BD_g_decam = data_bd_temp.columns[1]
BD_r_decam = data_bd_temp.columns[2]
BD_i_decam = data_bd_temp.columns[3]
BD_z_decam = data_bd_temp.columns[4]
BD_Y_vhs = data_bd_temp.columns[5]
BD_J_vhs = data_bd_temp.columns[6]
BD_H_vhs = data_bd_temp.columns[7]
BD_K_vhs = data_bd_temp.columns[8]
BD_W1 = data_bd_temp.columns[9]
BD_W2 = data_bd_temp.columns[10]
BD1_vec_flux = np.array([BD_g_decam[0], BD_r_decam[0], BD_i_decam[0], BD_z_decam[0], BD_Y_vhs[0], BD_J_vhs[0], BD_H_vhs[0], BD_K_vhs[0], BD_W1[0], BD_W2[0]])
BD2_vec_flux = np.array([BD_g_decam[1], BD_r_decam[1], BD_i_decam[1], BD_z_decam[1], BD_Y_vhs[1], BD_J_vhs[1], BD_H_vhs[1], BD_K_vhs[1], BD_W1[1], BD_W2[1]])
BD3_vec_flux = np.array([BD_g_decam[2], BD_r_decam[2], BD_i_decam[2], BD_z_decam[2], BD_Y_vhs[2], BD_J_vhs[2], BD_H_vhs[2], BD_K_vhs[2], BD_W1[2], BD_W2[2]])
BD4_vec_flux = np.array([BD_g_decam[3], BD_r_decam[3], BD_i_decam[3], BD_z_decam[3], BD_Y_vhs[3], BD_J_vhs[3], BD_H_vhs[3], BD_K_vhs[3], BD_W1[3], BD_W2[3]])
BD_all_vec_flux = np.array([BD1_vec_flux, BD2_vec_flux, BD3_vec_flux, BD4_vec_flux])
print("------------------------------------------ BD1 Template----------------------------------------------------")
print(BD_all_vec_flux[0])

# IIb : Make vector arrays for Quasars templates
QSO_g_decam = data_qso_temp.columns[2]
QSO_r_decam = data_qso_temp.columns[3]
QSO_i_decam = data_qso_temp.columns[4]
QSO_z_decam = data_qso_temp.columns[5]
QSO_Y_vhs = data_qso_temp.columns[6]
QSO_J_vhs = data_qso_temp.columns[7]
QSO_H_vhs = data_qso_temp.columns[8]
QSO_K_vhs = data_qso_temp.columns[9]
QSO_W1 = data_qso_temp.columns[10]
QSO_W2 = data_qso_temp.columns[11]

QSO_vec_flux = []
for i in range(len(data_qso_temp)):
    vec_flux = [QSO_g_decam[i], QSO_r_decam[i], QSO_i_decam[i], QSO_z_decam[i], QSO_Y_vhs[i], QSO_J_vhs[i], QSO_H_vhs[i], QSO_K_vhs[i], QSO_W1[i], QSO_W2[i]]
    QSO_vec_flux.append(vec_flux)

QSO_all_vec_flux = np.array(QSO_vec_flux)
print("------------------------------------------ QSO1 Template----------------------------------------------------")
print(QSO_all_vec_flux[0])


# III : Define a scaling and Chi2 functions
def a_scale(vec_flux_obs, vec_fluxe_obs, vec_flux_model):
    # ---- Obtain scaling factor for Chi2
    a = np.sum((vec_flux_obs * vec_flux_model) / (vec_fluxe_obs) ** 2) / np.sum(
        (vec_flux_model) ** 2 / (vec_fluxe_obs) ** 2)
    return a


def chi2_calc(vec_flux_obs, vec_fluxe_obs, vec_flux_model, a):
    # ---- Obtain scaling factor for Chi2
    chi2 = np.sum((vec_flux_obs - a * vec_flux_model) ** 2 / (vec_fluxe_obs) ** 2)
    return chi2

# IV : Read wavebands, fluxes and errors from source file
header = list(data_fil.columns)
select_err = "err"
del header[:4]
errors = [i for i in header if select_err in i]
wavebands = [i for i in header if i not in errors]
print("-----------------------------------------WAVEBANDS ------------------------------------------------------")
print(wavebands)

# a. Make array for central fluxes of the bands(each of them)
vec_flux = data_fil[["g_prime_delve", "r_prime_delve", "i_prime_delve", "z_prime_delve", "vista.vircam.Y_vhs", "vista.vircam.J_vhs", "vista.vircam.H_vhs", "vista.vircam.Ks_vhs", "WISE1", "WISE2"]].copy()
print("-----------------------------------------FLUXES ------------------------------------------------------")
vec_flux_data = np.delete(vec_flux, (0), axis=0)
# vec_flux_row = vec_flux[0]
# vec_flux_obs = np.array([vec_flux_row[0], vec_flux_row[1], vec_flux_row[2],vec_flux_row[3],vec_flux_row[4],vec_flux_row[5],vec_flux_row[6], vec_flux_row[7], vec_flux_row[8], vec_flux_row[9]])
print(vec_flux_data)

# b. Make another array for errors of the bands(each of them)
vec_fluxe = data_fil[["g_prime_err_delve", "r_prime_err_delve", "i_prime_err_delve", "z_prime_err_delve", "vista.vircam.Y_err_vhs", "vista.vircam.J_err_vhs", "vista.vircam.H_err_vhs", "vista.vircam.Ks_err_vhs", "WISE1_err", "WISE2_err"]].copy()
print("------------------------------------------ WAVEBAND ERRORS----------------------------------------------------")
print(errors)
print("------------------------------------------ FLUX ERRORS----------------------------------------------------")
vec_fluxe_data = np.delete(vec_fluxe, (0), axis=0)
# vec_fluxe_row = vec_fluxe[1]
# vec_fluxe_obs = np.array([vec_fluxe_row[0], vec_fluxe_row[1], vec_fluxe_row[2],vec_fluxe_row[3],vec_fluxe_row[4],vec_fluxe_row[5],vec_fluxe_row[6], vec_fluxe_row[7], vec_fluxe_row[8], vec_fluxe_row[9]])
print(vec_fluxe_data)

# c. Make all arrays with the same dtype in this case float
# vec_flux_obs = vec_flux_obs.astype(float)
# vec_fluxe_obs = vec_fluxe_obs.astype(float)
vec_flux_model_BD = BD_all_vec_flux.astype(float)
vec_flux_model_QSO = QSO_all_vec_flux.astype(float)

# Trying to make an array without nan values
# flux_array = np.array(vec_flux)
# flux_array[np.isnan(flux_array)] = 0
# print(flux_array)


# IV : Make empty arrays for Chi2 results of Brown Dwarfs and Quasars and calculate it.
BD_Chi2_array = []
QSO_Chi2_array = []
for i in range(len(data_fil)):
    vec_flux_row = vec_flux[i]
    vec_fluxe_row = vec_fluxe[i]
    vec_flux_obs = np.array([vec_flux_row[0], vec_flux_row[1], vec_flux_row[2], vec_flux_row[3], vec_flux_row[4], vec_flux_row[5], vec_flux_row[6], vec_flux_row[7], vec_flux_row[8], vec_flux_row[9]])
    vec_fluxe_obs = np.array([vec_fluxe_row[0], vec_fluxe_row[1], vec_fluxe_row[2],vec_fluxe_row[3],vec_fluxe_row[4],vec_fluxe_row[5],vec_fluxe_row[6], vec_fluxe_row[7], vec_fluxe_row[8], vec_fluxe_row[9]])
# a. Calculate scaling factor and Chi2 for BDs
    a_BD = a_scale(vec_flux_obs, vec_fluxe_obs, vec_flux_model_BD)
    a_BD = a_BD.astype(float)
    Chi2_BD = chi2_calc(vec_flux_obs, vec_fluxe_obs, vec_flux_model_BD, a_BD)
    BD_Chi2_array.append(Chi2_BD)
# b. Calculate scaling factor and Chi2 for QSOs
    a_QSO = a_scale(vec_flux_obs, vec_fluxe_obs, vec_flux_model_QSO)
    a_QSO = a_QSO.astype(float)
    Chi2_QSO = chi2_calc(vec_flux_obs, vec_fluxe_obs, vec_flux_model_QSO, a_QSO)
    QSO_Chi2_array.append(Chi2_QSO)

BD_chi2 = np.array(BD_Chi2_array)
QSO_chi2 = np.array(QSO_Chi2_array)
print("===============================================")
print("Chi2 BD:", BD_chi2)
print("Chi2 QSO:", QSO_chi2)

# # ----Plots
# plt.plot(ll_vec_red, vec_flux_model_BD * a_BD, 'o', color='k')
# plt.plot(ll_vec_red, vec_flux_model_QSO * a_QSO, 'o', color='b')
# plt.errorbar(ll_vec_red, vec_flux_obs, yerr=vec_fluxe_obs, fmt='o', color='r')
# # plt.plot(bdRA_l, bdRA_f1 * a_BD, color='gray')
# plt.plot(qso_S15_wave * (1 + dQSO_red), qso_S15_flux * a_QSO, color='b')
# plt.savefig("plot.png")
