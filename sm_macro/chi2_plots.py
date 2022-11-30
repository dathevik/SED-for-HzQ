import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import Table
from matplotlib.pyplot import figure

from synphot import SourceSpectrum, units, SpectralElement, etau_madau, Observation
from synphot.models import Empirical1D, BlackBodyNorm1D, ConstFlux1D

#------------------------------- Define functions ---------------------------------
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

#------------------------------- Main code ---------------------------------
# I: Define the path and print the data inside
filters_path = r'/home/dathev/PhD_Projects/4MOST/chicode_sm/cigale_flux_ALL-SKY_delve.dat'
BD_temp_path = r'/home/dathev/PhD_Projects/4MOST/chicode_sm/BDspectra/BDRA_fluxes_mJy.dat'
QSO_temp_path = r'/home/dathev/PhD_Projects/4MOST/chicode_sm/temp_qso/Selsing2015_fluxes_mJy.dat'
QSO_spec_path = r'/home/dathev/PhD_Projects/4MOST/chicode_sm/temp_qso/Selsing2015.dat'
source_path = r'/home/dathev/PhD_Projects/4MOST/chicode_sm/BDspectra/LDwarf/spex-prism_2MASSIJ0103320+193536_20030919_CRU04A.txt'
bdRA_path = '/home/dathev/PhD_Projects/4MOST/chicode_sm/BDspectra/RAssef/'

#--- CM: my path that I define to run this code in my laptop
# filters_path = r'/mnt/Data/Tatevik/4most/tables/Victoria_full/cigale_flux_ALL-SKY_delve.dat'
# BD_temp_path = r'/mnt/Data/Tatevik/chicode_sm/BDspectra/RAssef/BDRA_fluxes_mJy.dat'
# QSO_temp_path = r'/mnt/Data/Tatevik/chicode_sm/temp_qso/Selsing2015_fluxes_mJy.dat'

# CM: It is good when one is writing and texting the code to print the entire files it is reading.
# However, I would later comment those "print" commands.
# This is to make the code a bit more agile, and especially when the data table will be huge.

#print("DATA INSIDE", BD_temp_path, "FILE")
print("READING BD:", BD_temp_path, "FILE")
data_bd_temp = ascii.read(BD_temp_path)
print(data_bd_temp)

print("READING QSOs:", QSO_temp_path, "FILE")
data_qso_temp = ascii.read(QSO_temp_path)
print(data_qso_temp)

print("READING INPUT CATALOG:", filters_path, "FILE")
data_fil = ascii.read(filters_path)
print(data_fil)

print("READING SELSING 2015:", QSO_spec_path, "FILE")
data_qso_spec = ascii.read(QSO_spec_path)
print(data_qso_spec)

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

# CM: Make list with BDs type
BD_type_vec = ["BD1","BD2","BD3","BD4"]

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
    vec_flux = [QSO_g_decam[i], QSO_r_decam[i], QSO_i_decam[i], QSO_z_decam[i], QSO_Y_vhs[i], QSO_J_vhs[i], QSO_H_vhs[i], QSO_K_vhs[i], QSO_W1[i], QSO_W2[i]]
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
vec_flux = data_fil[["g_prime_delve", "r_prime_delve", "i_prime_delve", "z_prime_delve", "vista.vircam.Y_vhs", "vista.vircam.J_vhs", "vista.vircam.H_vhs", "vista.vircam.Ks_vhs", "WISE1", "WISE2"]].copy()
print("-----------------------------------------FLUXES ------------------------------------------------------")
vec_flux_data = np.delete(vec_flux, (0), axis=0)
#---- CM: Same result could be obtained by doing :
#vec_flux_data = np.array(vec_flux)
print(vec_flux_data)

# b. Make another array for errors of the bands(each of them)
vec_fluxe = data_fil[["g_prime_err_delve", "r_prime_err_delve", "i_prime_err_delve", "z_prime_err_delve", "vista.vircam.Y_err_vhs", "vista.vircam.J_err_vhs", "vista.vircam.H_err_vhs", "vista.vircam.Ks_err_vhs", "WISE1_err", "WISE2_err"]].copy()
print("------------------------------------------ WAVEBAND ERRORS----------------------------------------------------")
print(errors)
print("------------------------------------------ FLUX ERRORS----------------------------------------------------")
vec_fluxe_data = np.delete(vec_fluxe, (0), axis=0)
#---- CM: Same result could be obtained by doing:
#vec_fluxe_data = np.array(vec_fluxe)
print(vec_fluxe_data)

# c. Make all arrays with the same dtype in this case float
# vec_flux_obs = vec_flux_obs.astype(float)
# vec_fluxe_obs = vec_fluxe_obs.astype(float)
vec_flux_model_BD = BD_all_vec_flux.astype(float)
vec_flux_model_QSO = QSO_all_vec_flux.astype(float)


# IV : Make empty arrays for Chi2 results of Brown Dwarfs and Quasars and calculate it.
for i in range(len(data_fil)):

    # CM: I add here a printout to see at which object the code is
    print("----------------------------------------- RUN FOR OBJECT NUMBER",i,"----------------------------------------")
    BD_Chi2_array = []
    QSO_Chi2_array = []
    R_Chi = []

    vec_flux_row = vec_flux[i]
    vec_fluxe_row = vec_fluxe[i]
    vec_flux_obs0 = np.array([vec_flux_row[0], vec_flux_row[1], vec_flux_row[2], vec_flux_row[3], vec_flux_row[4], vec_flux_row[5], vec_flux_row[6], vec_flux_row[7], vec_flux_row[8], vec_flux_row[9]])
    vec_fluxe_obs0 = np.array([vec_fluxe_row[0], vec_fluxe_row[1], vec_fluxe_row[2],vec_fluxe_row[3],vec_fluxe_row[4],vec_fluxe_row[5],vec_fluxe_row[6], vec_fluxe_row[7], vec_fluxe_row[8], vec_fluxe_row[9]])
    
    #---- CM: mask the nan value, which means it will not be used for the calculation
    mask_nan = ~np.isnan(vec_flux_obs0)
    vec_flux_obs = vec_flux_obs0[mask_nan]
    vec_fluxe_obs = vec_fluxe_obs0[mask_nan]
    #print(vec_flux_obs)

    # CM: Here I added another (sub)loop on the BDs templates
    # a. Calculate scaling factor and Chi2 for BDs    
    for j in range(len(data_bd_temp)):
       vec_flux_model_BD_j0 = vec_flux_model_BD[j]
       vec_flux_model_BD_j  = vec_flux_model_BD_j0[mask_nan] 
       a_BD = a_scale(vec_flux_obs, vec_fluxe_obs, vec_flux_model_BD_j)
       a_BD = a_BD.astype(float)
       Chi2_BD = chi2_calc(vec_flux_obs, vec_fluxe_obs, vec_flux_model_BD_j, a_BD)
       BD_Chi2_array.append(Chi2_BD)
    print(BD_Chi2_array)
    
    # CM: Here I added another (sub)loop on the QSOs templates
    # b. Calculate scaling factor and Chi2 for QSOs
    for k in range(len(data_qso_temp)):
       vec_flux_model_QSO_k0 = vec_flux_model_QSO[k]
       vec_flux_model_QSO_k  = vec_flux_model_QSO_k0[mask_nan]     
       a_QSO = a_scale(vec_flux_obs, vec_fluxe_obs, vec_flux_model_QSO_k)
       a_QSO = a_QSO.astype(float)
       Chi2_QSO = chi2_calc(vec_flux_obs, vec_fluxe_obs, vec_flux_model_QSO_k, a_QSO)
       QSO_Chi2_array.append(Chi2_QSO)
    print(QSO_Chi2_array)

    # CM : Here I add the part where it calculates and print out the template with the best Chi2 for QSOs and BDs
    # c1. Calculate the lowest value for the Chi2 for the BDs
    BD_Chi2_min        = np.min(BD_Chi2_array)    #-- minimum value
    BD_Chi2_min_ind    = np.argmin(BD_Chi2_array) #-- position in array
    BD_Chi2_min_temp   = BD_type_vec[BD_Chi2_min_ind]  #-- corresponding best template
    print("-------------------------------------------------")
    print("Best Chi2 BD:", BD_Chi2_min)
    print("Which BD templates:", BD_Chi2_min_temp)


    # c2. Calculate the lowest value for the Chi2 for the QSOs
    QSO_Chi2_min = np.min(QSO_Chi2_array)   #-- minimum value
    QSO_Chi2_min_ind  = np.argmin(QSO_Chi2_array) #-- position in array
    QSO_Chi2_min_z    = QSO_z_vec[QSO_Chi2_min_ind] #-- corresponding best template -> best redshift
    print("-------------------------------------------------")
    print("Best Chi2 QSO:", QSO_Chi2_min)
    print("Which QSO redshift:", QSO_Chi2_min_z)

# Calculating ratio of chi2 of each QSO and BD templates
    R = BD_Chi2_min/QSO_Chi2_min
    R_Chi.append(R)
    print("-------------------------------------------------")
    print("The Chi2 ratio is", R)



#---- CM: This will be the for the last object from the catalog only, so I comment this out because I do not think it will be useful for now!
#BD_chi2 = np.array(BD_Chi2_array)
#QSO_chi2 = np.array(QSO_Chi2_array)
#print("===============================================")
#print("Chi2 BD:", BD_chi2)
#print("Chi2 QSO:", QSO_chi2)

vec_flux_model_BD_best = BD_all_vec_flux[BD_Chi2_min_ind][mask_nan] * a_BD  # not this BD but Roberto's templates
vec_flux_model_QSO_best = QSO_all_vec_flux[QSO_Chi2_min_ind][mask_nan] * a_QSO

# # ----Plots

# --------RAssef BD template
bdRA_name = bdRA_path+'lrt_a07_BDs.dat'
bdRA_data = Table(ascii.read(bdRA_name))
bdRA_l1 = bdRA_data.columns[0]*10**4 #--AA
bdRA_l2 = bdRA_data.columns[1]*10**4 #--AA
bdRA_l = bdRA_l2 + (bdRA_l1 - bdRA_l2)/2. #---bin center in the template

#----- Fluxes vectors in nuFnu (Hz Jy)
bdRA_nf1 = bdRA_data.columns[2]
bdRA_nf2 = bdRA_data.columns[3]
bdRA_nf3 = bdRA_data.columns[4]
bdRA_nf4 = bdRA_data.columns[5]

#----- Convert fluxes from nuFnu (Hz Jy) in (erg/s/cm2/A)
bdRA_f1 = bdRA_nf1 / bdRA_l
bdRA_f2 = bdRA_nf2 / bdRA_l
bdRA_f3 = bdRA_nf3 / bdRA_l
bdRA_f4 = bdRA_nf4 / bdRA_l

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

# Plotting the RAssef BD best model
bdRA_f_best = (bdRA_data.columns[2 + int(BD_Chi2_min_ind)] / bdRA_l) * a_BD
bdRA_fmJy_best = flux_from_cgs_to_mJy(bdRA_f_best, bdRA_l)
plt.plot(bdRA_l, bdRA_fmJy_best) # add for quasars
plt.savefig("RAssef_BD.png", dpi=160)
plt.close()

# Plotting the QSO best model
w_rest = np.arange(900.,75000,10)
spec_f = np.array(data_qso_spec)
wave = data_qso_spec.columns[0]
flux = data_qso_spec.columns[1] * a_QSO
sp = SourceSpectrum(Empirical1D, points=wave, lookup_table=flux, keep_neg=True)
sp.plot()
plt.savefig("Selsing_QSO.png", dpi=160)
plt.close()

# Redshifting the QSO best model
redshift = QSO_Chi2_min_z
sp_z_chi2 = SourceSpectrum(sp.model, z=redshift, z_type='conserve_flux')
plt.plot(wave, sp(wave), 'grey', label='rest frame')
plt.plot(wave, sp_z_chi2(wave), 'b', label=f'z={QSO_Chi2_min_z}')
plt.legend(loc='upper right')
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel('Flux (not normalized)')
plt.savefig("Selsing_z.png", dpi=160)
plt.close()


# ---Apply the absorption from the Intergalactic medium
extcurve_best = etau_madau(w_rest, redshift)
sp_ext_best = sp_z_chi2 * extcurve_best
flux_mJY_best = flux_from_cgs_to_mJy(sp_ext_best(w_rest), w_rest)
plt.plot(w_rest, sp(w_rest), 'grey', label='rest frame')
plt.plot(w_rest, flux_mJY_best, 'b', label=f'z={QSO_Chi2_min_z}')
plt.legend(loc='upper right')
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel('Flux (not normalized)')
plt.title("IGM Absoprtion")
plt.savefig("IGM.png", dpi=160)
plt.close()


# Plotting the best BD and QSO models
plt.scatter(ll_vec_best, vec_flux_model_BD_best, color="c", label=f"Best BD template ({BD_Chi2_min_temp})")
plt.scatter(ll_vec_best, vec_flux_model_QSO_best, color="m", label=f"Best QSO template (z={QSO_Chi2_min_z})")
plt.errorbar(ll_vec_best, vec_flux_obs, yerr=vec_fluxe_obs, fmt='o', color='b', label="Observed spectra with errors")
plt.legend(loc="upper right")
plt.xlabel("Central Wavelength (Å)")
plt.ylabel("Flux (mJy)")
plt.savefig("best_models.png", dpi=160)
plt.close()


# Plotting the best BD and QSO models with the IGM spectrum
plt.plot(bdRA_l, bdRA_fmJy_best, color="y",  label=f'Brown Dwarf spectrum with {BD_Chi2_min_temp}')
plt.plot(w_rest, flux_mJY_best, color="g", label=f'Quasar spectrum with z={QSO_Chi2_min_z}')
plt.scatter(ll_vec_best, vec_flux_model_BD_best, color="c", label=f"Best BD template ({BD_Chi2_min_temp})")
plt.scatter(ll_vec_best, vec_flux_model_QSO_best, color="m", label=f"Best QSO template (z={QSO_Chi2_min_z})")
plt.errorbar(ll_vec_best, vec_flux_obs, yerr=vec_fluxe_obs, fmt='o', color='b', label="Observed spectra with errors")
plt.legend(loc="upper right")
plt.xlabel("Central Wavelength (Å)")
plt.ylabel("Flux (mJy)")
plt.savefig("BD_QSO_best_models.png", dpi=160)
plt.close
