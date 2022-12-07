import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import Table
from synphot import SourceSpectrum, units, SpectralElement, etau_madau, Observation
from synphot.models import Empirical1D, BlackBodyNorm1D, ConstFlux1D


# ------------------------------- Define functions ---------------------------------
# CM: Writing a code, I would put first at the beginning of the code, the definition of functions (in our case now a_scale, chi2_scales),
#     instead of having them in the flow of the code.
#     In this way it will be easier to locate them, and the different parts of the code are divided.
def a_scale(vec_flux_obs, vec_fluxe_obs, vec_flux_model):
    # ---- Obtain scaling factor for Chi2
    a = np.sum((vec_flux_obs * vec_flux_model) / (vec_fluxe_obs) ** 2) / np.sum(
        (vec_flux_model) ** 2 / (vec_fluxe_obs) ** 2)
    return a


def chi2_calc(vec_flux_obs, vec_fluxe_obs, vec_flux_model, a):
    # ---- Obtain scaling factor for Chi2
    chi2 = np.sum((vec_flux_obs - a * vec_flux_model) ** 2 / (vec_fluxe_obs) ** 2)
    return chi2

def flux_from_cgs_to_mJy(flux_cgs,ll):
  #---- Obtain flux in mJy from flux in erg/s/cm2/A and considering effective wavelength of the filter in AA
  flux_mJy = (3.33564095E+04 * flux_cgs * ll**2) *1E-3
  return flux_mJy

# ------------------------------- Main code ---------------------------------
# I: Define the path and print the data inside
filters_path = r'/home/dathev/PhD_Projects/4MOST/chicode_sm/cigale_flux_ALL-SKY_delve.dat'
# BD_temp_path = r'/home/dathev/PhD_Projects/4MOST/chicode_sm/BDspectra/BDRA_fluxes_mJy.dat'
QSO_temp_path = r'/home/dathev/PhD_Projects/4MOST/chicode_sm/temp_qso/Selsing2015_fluxes_mJy.dat'
bdRA_path = '/home/dathev/PhD_Projects/4MOST/chicode_sm/BDspectra/RAssef/'
BD_temp_path = '/home/dathev/PhD_Projects/4MOST/chicode_sm/BDspectra/RAssef/lrt_a07_BDs.dat'
# --- CM: my path that I define to run this code in my laptop
# filters_path = r'/mnt/Data/Tatevik/4most/tables/Victoria_full/cigale_flux_ALL-SKY_delve.dat'
# BD_temp_path = r'/mnt/Data/Tatevik/chicode_sm/BDspectra/RAssef/BDRA_fluxes_mJy.dat'
# QSO_temp_path = r'/mnt/Data/Tatevik/chicode_sm/temp_qso/Selsing2015_fluxes_mJy.dat'

# CM: It is good when one is writing and texting the code to print the entire files it is reading.
# However, I would later comment those "print" commands.
# This is to make the code a bit more agile, and especially when the data table will be huge.

# print("DATA INSIDE", BD_temp_path, "FILE")
print("READING BD:", BD_temp_path, "FILE")
data_bd_temp = ascii.read(BD_temp_path)
print(data_bd_temp)

print("READING QSOs:", QSO_temp_path, "FILE")
data_qso_temp = ascii.read(QSO_temp_path)
print(data_qso_temp)

print("READING INPUT CATALOG:", filters_path, "FILE")
data_fil = ascii.read(filters_path)
print(data_fil)

# IIa : Make vector arrays for Brown Dwarf templates
BD_data = Table(ascii.read(BD_temp_path))
BD_l1 = BD_data.columns[0]*10**4 #--AA
BD_l2 = BD_data.columns[1]*10**4 #--AA
BD_l = BD_l2 + (BD_l1 - BD_l2)/2. #---bin center in the template

#----- Fluxes vectors in nuFnu (Hz Jy)
BD_nf1 = BD_data.columns[2]
BD_nf2 = BD_data.columns[3]
BD_nf3 = BD_data.columns[4]
BD_nf4 = BD_data.columns[5]

#----- Convert fluxes from nuFnu (Hz Jy) in (erg/s/cm2/A)
BD1_vec_flux = np.array(BD_nf1 / BD_l)
BD2_vec_flux = np.array(BD_nf2 / BD_l)
BD3_vec_flux = np.array(BD_nf3 / BD_l)
BD4_vec_flux = np.array(BD_nf4 / BD_l)
# CM: Make list with BDs type
BD_type_vec = ["BD1", "BD2", "BD3", "BD4"]

BD1_flux = []
BD2_flux = []
BD3_flux = []
BD4_flux = []
for i in range(0, 10):
    vec_flux_BD1 = BD1_vec_flux[i]
    BD1_flux.append(vec_flux_BD1)
    vec_flux_BD2 = BD2_vec_flux[i]
    BD2_flux.append(vec_flux_BD2)
    vec_flux_BD3 = BD3_vec_flux[i]
    BD3_flux.append(vec_flux_BD3)
    vec_flux_BD4 = BD4_vec_flux[i]
    BD4_flux.append(vec_flux_BD4)

BD_all_vec_flux = np.array([BD1_flux, BD2_flux, BD3_flux, BD4_flux])

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

# CM: Read Column of Redshift
QSO_z_vec = data_qso_temp.columns[1]

QSO_vec_flux = []
for i in range(len(data_qso_temp)):
    vec_flux = [QSO_g_decam[i], QSO_r_decam[i], QSO_i_decam[i], QSO_z_decam[i], QSO_Y_vhs[i], QSO_J_vhs[i],
                QSO_H_vhs[i], QSO_K_vhs[i], QSO_W1[i], QSO_W2[i]]
    QSO_vec_flux.append(vec_flux)

QSO_all_vec_flux = np.array(QSO_vec_flux)
print("------------------------------------------ QSO1 Template----------------------------------------------------")
print(QSO_all_vec_flux[0])

# IV : Read wavebands, fluxes and errors from source file
header = list(data_fil.columns)
select_err = "err"
del header[:4]
errors = [i for i in header if select_err in i]
wavebands = [i for i in header if i not in errors]
print("-----------------------------------------WAVEBANDS ------------------------------------------------------")
print(wavebands)

# a. Make array for central fluxes of the bands(each of them)
vec_flux = data_fil[
    ["g_prime_delve", "r_prime_delve", "i_prime_delve", "z_prime_delve", "vista.vircam.Y_vhs", "vista.vircam.J_vhs",
     "vista.vircam.H_vhs", "vista.vircam.Ks_vhs", "WISE1", "WISE2"]].copy()
print("-----------------------------------------FLUXES ------------------------------------------------------")
vec_flux_data = np.delete(vec_flux, (0), axis=0)
# ---- CM: Same result could be obtained by doing :
# vec_flux_data = np.array(vec_flux)
print(vec_flux_data)

# b. Make another array for errors of the bands(each of them)
vec_fluxe = data_fil[
    ["g_prime_err_delve", "r_prime_err_delve", "i_prime_err_delve", "z_prime_err_delve", "vista.vircam.Y_err_vhs",
     "vista.vircam.J_err_vhs", "vista.vircam.H_err_vhs", "vista.vircam.Ks_err_vhs", "WISE1_err", "WISE2_err"]].copy()
print("------------------------------------------ WAVEBAND ERRORS----------------------------------------------------")
print(errors)
print("------------------------------------------ FLUX ERRORS----------------------------------------------------")
vec_fluxe_data = np.delete(vec_fluxe, (0), axis=0)
# ---- CM: Same result could be obtained by doing:
# vec_fluxe_data = np.array(vec_fluxe)
print(vec_fluxe_data)

# c. Make all arrays with the same dtype in this case float
# vec_flux_obs = vec_flux_obs.astype(float)
# vec_fluxe_obs = vec_fluxe_obs.astype(float)
vec_flux_model_BD = BD_all_vec_flux.astype(float)
vec_flux_model_QSO = QSO_all_vec_flux.astype(float)
# IV : Make empty arrays for Chi2 results of Brown Dwarfs and Quasars and calculate it.
for i in range(len(data_fil)):

    # CM: I add here a printout to see at which object the code is
    print("----------------------------------------- RUN FOR OBJECT NUMBER", i,
          "----------------------------------------")
    BD_Chi2_array = []
    QSO_Chi2_array = []

    vec_flux_row = vec_flux[i]
    vec_fluxe_row = vec_fluxe[i]
    vec_flux_obs0 = np.array(
        [vec_flux_row[0], vec_flux_row[1], vec_flux_row[2], vec_flux_row[3], vec_flux_row[4], vec_flux_row[5],
         vec_flux_row[6], vec_flux_row[7], vec_flux_row[8], vec_flux_row[9]])
    vec_fluxe_obs0 = np.array(
        [vec_fluxe_row[0], vec_fluxe_row[1], vec_fluxe_row[2], vec_fluxe_row[3], vec_fluxe_row[4], vec_fluxe_row[5],
         vec_fluxe_row[6], vec_fluxe_row[7], vec_fluxe_row[8], vec_fluxe_row[9]])

    # ---- CM: mask the nan value, which means it will not be used for the calculation
    mask_nan = ~np.isnan(vec_flux_obs0)
    vec_flux_obs = vec_flux_obs0[mask_nan]
    vec_fluxe_obs = vec_fluxe_obs0[mask_nan]
    # print(vec_flux_obs)
    # CM: Here I added another (sub)loop on the BDs templates
    # a. Calculate scaling factor and Chi2 for BDs
    for j in range(len(vec_flux_model_BD)):
        vec_flux_model_BD_j0 = vec_flux_model_BD[j]
        vec_flux_model_BD_j = vec_flux_model_BD_j0[mask_nan]
            # vec_flux_model_BD_j0 = vec_flux_model_BD[j]
            # vec_flux_model_BD_j = vec_flux_model_BD_j0
        a_BD = a_scale(vec_flux_obs, vec_fluxe_obs, vec_flux_model_BD_j)
        a_BD = a_BD.astype(float)
        Chi2_BD = chi2_calc(vec_flux_obs, vec_fluxe_obs, vec_flux_model_BD_j, a_BD)
        BD_Chi2_array.append(Chi2_BD)
    print(BD_Chi2_array)

    # CM: Here I added another (sub)loop on the QSOs templates
    # b. Calculate scaling factor and Chi2 for QSOs
    for k in range(len(data_qso_temp)):
        vec_flux_model_QSO_k0 = vec_flux_model_QSO[k]
        vec_flux_model_QSO_k = vec_flux_model_QSO_k0[mask_nan]
        a_QSO = a_scale(vec_flux_obs, vec_fluxe_obs, vec_flux_model_QSO_k)
        a_QSO = a_QSO.astype(float)
        Chi2_QSO = chi2_calc(vec_flux_obs, vec_fluxe_obs, vec_flux_model_QSO_k, a_QSO)
        QSO_Chi2_array.append(Chi2_QSO)
    print(QSO_Chi2_array)

    # CM : Here I add the part where it calculates and print out the template with the best Chi2 for QSOs and BDs
    # c1. Calculate the lowest value for the Chi2 for the BDs
    BD_Chi2_min = np.min(BD_Chi2_array)  # -- minimum value
    BD_Chi2_min_ind = np.argmin(BD_Chi2_array)  # -- position in array
    BD_Chi2_min_temp = BD_type_vec[BD_Chi2_min_ind]  # -- corresponding best template
    print("-------------------------------------------------")
    print("Best Chi2 BD:", BD_Chi2_min)
    print("Which BD templates:", BD_Chi2_min_temp)

    # c2. Calculate the lowest value for the Chi2 for the QSOs
    QSO_Chi2_min = np.min(QSO_Chi2_array)  # -- minimum value
    QSO_Chi2_min_ind = np.argmin(QSO_Chi2_array)  # -- position in array
    QSO_Chi2_min_z = QSO_z_vec[QSO_Chi2_min_ind]  # -- corresponding best template -> best redshift
    print("-------------------------------------------------")
    print("Best Chi2 QSO:", QSO_Chi2_min)
    print("Which QSO redshift:", QSO_Chi2_min_z)

# ---- CM: This will be the for the last object from the catalog only, so I comment this out because I do not think it will be useful for now!
# BD_chi2 = np.array(BD_Chi2_array)
# QSO_chi2 = np.array(QSO_Chi2_array)
# print("===============================================")
# print("Chi2 BD:", BD_chi2)
# print("Chi2 QSO:", QSO_chi2)

vec_flux_model_BD_best = BD_all_vec_flux[BD_Chi2_min_ind][mask_nan] * a_BD  # not this BD but Roberto's templates
vec_flux_model_QSO_best = QSO_all_vec_flux[QSO_Chi2_min_ind][mask_nan] * a_QSO

# # ----Plots
#-----Define central wavelength for plot it later for check
ll_g = 4798.3527009231575 #Angstrom
ll_r = 6407.493598028656 #Angstrom
ll_i = 7802.488114833454 #Angstrom
ll_z = 9144.625340022629 #Angstrom
ll_J = 12325.125694338809 #Angstrom
ll_Y = 10201.359507821942 #Angstrom
ll_H = 16473.95843628733 #Angstrom
ll_K = 22045.772662096875 #Angstrom
ll_W1 = 33791.878497259444 #Angstrom
ll_W2 = 46292.93969033106 #Angstrom
ll_vec = np.array([ll_g,ll_r,ll_i,ll_z,ll_Y,ll_J,ll_H,ll_K,ll_W1,ll_W2])
ll_vec_best = ll_vec[mask_nan] #---remember the mask, otherwise the two vectors will not match in length!!
plt.plot(ll_vec_best, vec_flux_model_BD_best)

# bdRA_fmJy_best = flux_from_cgs_to_mJy(BD_all_vec_flux, bdRA_l)
bdRA_fmJy_best = flux_from_cgs_to_mJy(vec_flux_model_BD_best, bdRA_l)
plt.plot(bdRA_l, bdRA_fmJy_best)
plt.savefig("chiara.png")
plt.close

# Plotting the spectrum
sp = SourceSpectrum(Empirical1D, points=bdRA_l, lookup_table=bdRA_f1, keep_neg=True)
sp.plot()
plt.savefig("spectrum_chi.png")
plt.close()

# Plotting with QSO best
sp.plot()
plt.scatter(ll_vec_best, vec_flux_model_QSO_best, color="r", label="QSO z=4.5")
plt.errorbar(ll_vec_best, vec_flux_obs, yerr=vec_fluxe_obs, fmt='o', color='r')
plt.savefig("QSO_best.png")
plt.close()


# Plotting with BD best
sp.plot()
plt.plot(ll_vec_best, vec_flux_model_BD_best)
plt.errorbar(ll_vec_best, vec_flux_obs, yerr=vec_fluxe_obs, fmt='o', color='r')
plt.savefig("BD_best.png")
plt.close()


# plt.errorbar(ll_vec_best, vec_flux_obs, yerr=vec_fluxe_obs, fmt='o', color='r')
# plt.scatter(ll_vec_best, vec_flux_model_QSO_best, color="r", label="QSO z=4.5")
