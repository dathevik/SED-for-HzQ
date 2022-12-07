import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import ascii
from astropy.table import Table

from synphot import SourceSpectrum, units, SpectralElement, etau_madau, Observation
from synphot.models import Empirical1D, BlackBodyNorm1D, ConstFlux1D


#------------------------------- Define functions ---------------------------------

# function for a_scale which is a scaling factor and takes as an input of observed flux, observed flux error of the object and the template model
def a_scale(vec_flux_obs, vec_fluxe_obs, vec_flux_model):
    # ---- Obtain scaling factor for Chi2
    a = np.sum((vec_flux_obs * vec_flux_model) / (vec_fluxe_obs) ** 2) / np.sum(
        (vec_flux_model) ** 2 / (vec_fluxe_obs) ** 2)
    return a

# function for calculation of chi2 statistical parameter and takes as an input of observed flux, observed flux error of the object and the template model and a scaling factor
def chi2_calc(vec_flux_obs, vec_fluxe_obs, vec_flux_model, a):
    # ---- Obtain scaling factor for Chi2
    chi2 = np.sum((vec_flux_obs - a * vec_flux_model) ** 2 / (vec_fluxe_obs) ** 2)
    return chi2

# function for converting cgs parameters to mJy as the whole script works with mJy units of fluxes
def flux_from_cgs_to_mJy(flux_cgs,ll):
  #---- Obtain flux in mJy from flux in erg/s/cm2/A and considering effective wavelength of the filter in AA
  flux_mJy = (3.33564095E+04 * flux_cgs * ll**2) *1E-3
  return flux_mJy

#------------------------------- Main code ---------------------------------
# I: Define the path and print the data inside
# filters_path = r'/home/dathev/PhD_Projects/4MOST/chicode_sm/sm_macro/input test/cigale_flux_ALL-SKY_delve.dat'
filters_path = os.path.abspath('input test/cigale_flux_ALL-SKY_delve.dat')
BD_temp_path = os.path.abspath('input test/BDRA_fluxes_mJy.dat')
QSO_temp_path = os.path.abspath('input test/Selsing2015_fluxes_mJy.dat')
QSO_spec_path = os.path.abspath('input test/Selsing2015.dat')
bdRA_path = os.path.abspath('input test/lrt_a07_BDs.dat')

#--- CM: my path that I define to run this code in my laptop
# filters_path = r'/mnt/Data/Tatevik/4most/tables/Victoria_full/cigale_flux_ALL-SKY_delve.dat'
# BD_temp_path = r'/mnt/Data/Tatevik/chicode_sm/BDspectra/RAssef/BDRA_fluxes_mJy.dat'
# QSO_temp_path = r'/mnt/Data/Tatevik/chicode_sm/temp_qso/Selsing2015_fluxes_mJy.dat'

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

# IIa : Make the vector arrays for Brown Dwarf templates
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

# Make list with BDs type and print first template as an example
BD_type_vec = ["BD1","BD2","BD3","BD4"]

print("------------------------------------------ BD1 Template----------------------------------------------------")
print(BD_all_vec_flux[0])

# -------- Read RAssef BD templates for plotting and comparing the results
# bdRA_name = bdRA_path+'lrt_a07_BDs.dat'
bdRA_data = Table(ascii.read(bdRA_path))
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

#----- Read Column of Redshift in a loop and print the first template as an example
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

# IVa : Make array for central fluxes of the bands(each of them)
vec_flux = data_fil[["g_prime_delve", "r_prime_delve", "i_prime_delve", "z_prime_delve", "vista.vircam.Y_vhs", "vista.vircam.J_vhs", "vista.vircam.H_vhs", "vista.vircam.Ks_vhs", "WISE1", "WISE2"]].copy()
print("-----------------------------------------FLUXES ------------------------------------------------------")
vec_flux_data = np.delete(vec_flux, (0), axis=0)
#---- CM: Same result could be obtained by doing :
#vec_flux_data = np.array(vec_flux)
print(vec_flux_data)

# IVb : Make array for errors of the bands(each of them)
vec_fluxe = data_fil[["g_prime_err_delve", "r_prime_err_delve", "i_prime_err_delve", "z_prime_err_delve", "vista.vircam.Y_err_vhs", "vista.vircam.J_err_vhs", "vista.vircam.H_err_vhs", "vista.vircam.Ks_err_vhs", "WISE1_err", "WISE2_err"]].copy()
print("------------------------------------------ WAVEBAND ERRORS----------------------------------------------------")
print(errors)
print("------------------------------------------ FLUX ERRORS----------------------------------------------------")
vec_fluxe_data = np.delete(vec_fluxe, (0), axis=0)
print(vec_fluxe_data)

# c. Make all arrays with the same dtype in this case float
# vec_flux_obs = vec_flux_obs.astype(float)
# vec_fluxe_obs = vec_fluxe_obs.astype(float)
vec_flux_model_BD = BD_all_vec_flux.astype(float)
vec_flux_model_QSO = QSO_all_vec_flux.astype(float)


# V : Calculate Chi2 for BD and QSO templates
for i in range(len(data_fil)):
    print("----------------------------------------- RUN FOR OBJECT NUMBER",i,"----------------------------------------")
    BD_Chi2_array = []
    QSO_Chi2_array = []
    R_Chi = []

    vec_flux_row = vec_flux[i]
    vec_fluxe_row = vec_fluxe[i]
    vec_flux_obs0 = np.array([vec_flux_row[0], vec_flux_row[1], vec_flux_row[2], vec_flux_row[3], vec_flux_row[4], vec_flux_row[5], vec_flux_row[6], vec_flux_row[7], vec_flux_row[8], vec_flux_row[9]])
    vec_fluxe_obs0 = np.array([vec_fluxe_row[0], vec_fluxe_row[1], vec_fluxe_row[2],vec_fluxe_row[3],vec_fluxe_row[4],vec_fluxe_row[5],vec_fluxe_row[6], vec_fluxe_row[7], vec_fluxe_row[8], vec_fluxe_row[9]])
    
    #---- mask the nan value, which means it will not be used for the calculation
    mask_nan = ~np.isnan(vec_flux_obs0)
    vec_flux_obs = vec_flux_obs0[mask_nan]
    vec_fluxe_obs = vec_fluxe_obs0[mask_nan]
    ll_vec_best = ll_vec[mask_nan]
    #print(vec_flux_obs)

    # ---- parameters for plotting in a loop
    w_rest = np.arange(900., 75000, 10)
    spec_f = np.array(data_qso_spec)
    wave = data_qso_spec.columns[0]
    flux = data_qso_spec.columns[1]
    sp = SourceSpectrum(Empirical1D, points=wave, lookup_table=flux, keep_neg=True)
    sp.plot()

# Va : Calculate scaling factor and Chi2 for BDs
    for j in range(len(data_bd_temp)):
       vec_flux_model_BD_j0 = vec_flux_model_BD[j]
       vec_flux_model_BD_j  = vec_flux_model_BD_j0[mask_nan] 
       a_BD = a_scale(vec_flux_obs, vec_fluxe_obs, vec_flux_model_BD_j)
       a_BD = a_BD.astype(float)
       Chi2_BD = chi2_calc(vec_flux_obs, vec_fluxe_obs, vec_flux_model_BD_j, a_BD)
       BD_Chi2_array.append(Chi2_BD)
    print(BD_Chi2_array)

# Vb : Calculate scaling factor and Chi2 for QSOs
    for k in range(len(data_qso_temp)):
       vec_flux_model_QSO_k0 = vec_flux_model_QSO[k]
       vec_flux_model_QSO_k  = vec_flux_model_QSO_k0[mask_nan]     
       a_QSO = a_scale(vec_flux_obs, vec_fluxe_obs, vec_flux_model_QSO_k)
       a_QSO = a_QSO.astype(float)
       Chi2_QSO = chi2_calc(vec_flux_obs, vec_fluxe_obs, vec_flux_model_QSO_k, a_QSO)
       QSO_Chi2_array.append(Chi2_QSO)
    print(QSO_Chi2_array)

# VI : Calculate and print out the template with the best (lowest) Chi2 for QSOs and BDs
# VIa : Calculate the lowest value for the Chi2 for the BDs
    BD_Chi2_min        = np.min(BD_Chi2_array)    #-- minimum value
    BD_Chi2_min_ind    = np.argmin(BD_Chi2_array) #-- position in array
    BD_Chi2_min_temp   = BD_type_vec[BD_Chi2_min_ind]  #-- corresponding best template
    vec_flux_model_BD_best = BD_all_vec_flux[BD_Chi2_min_ind][mask_nan] * a_BD  # not this BD but Roberto's templates
    print("-------------------------------------------------")
    print("Best Chi2 BD:", BD_Chi2_min)
    print("Which BD templates:", BD_Chi2_min_temp)


# VIb. Calculate the lowest value for the Chi2 for the QSOs
    QSO_Chi2_min = np.min(QSO_Chi2_array)   #-- minimum value
    QSO_Chi2_min_ind  = np.argmin(QSO_Chi2_array) #-- position in array
    QSO_Chi2_min_z    = QSO_z_vec[QSO_Chi2_min_ind] #-- corresponding best template -> best redshift
    vec_flux_model_QSO_best = QSO_all_vec_flux[QSO_Chi2_min_ind][mask_nan] * a_QSO
    print("-------------------------------------------------")
    print("Best Chi2 QSO:", QSO_Chi2_min)
    print("Which QSO redshift:", QSO_Chi2_min_z)

# ----Calculate the ratio of chi2 of each QSO and BD templates
    R = BD_Chi2_min/QSO_Chi2_min
    R_Chi.append(R)
    print("-------------------------------------------------")
    print("The Chi2 ratio is", R)

# VII: Plot the results
# ---- Plot the best QSO template shifted with amount of redshift corresponding to the template
    redshift = QSO_Chi2_min_z
    sp_z_chi2 = SourceSpectrum(sp.model, z=redshift)
    plt.plot(wave, sp(wave), 'grey', label='rest frame')
    plt.plot(wave, sp_z_chi2(wave), 'b', label=f'z={QSO_Chi2_min_z}')
    plt.legend(loc='upper right')
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Flux (not normalized)')
    plt.savefig(f"output_test/{i}Selsing_z.png", dpi=160)
    plt.close()

# ---- Apply the absorption from the Intergalactic medium
    extcurve_best = etau_madau(w_rest, redshift)
    sp_ext_best = sp_z_chi2 * extcurve_best
    flux_mJY_best = flux_from_cgs_to_mJy(sp_ext_best(w_rest), w_rest) * a_QSO
    plt.plot(w_rest, sp(w_rest), 'grey', label='rest frame')
    plt.plot(w_rest, flux_mJY_best, 'b', label=f'z={QSO_Chi2_min_z}')
    plt.legend(loc='upper right')
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Flux (not normalized)')
    plt.title("IGM Absoprtion")
    plt.savefig(f"output_test/{i}_IGM.png", dpi=160)
    plt.close()

# ---- Plot the best BD and QSO models together and the source points with errors
    plt.scatter(ll_vec_best, vec_flux_model_BD_best, color="#836853", label=f"Best BD template ({BD_Chi2_min_temp})")
    plt.scatter(ll_vec_best, vec_flux_model_QSO_best, color="#004987", label=f"Best QSO template (z={QSO_Chi2_min_z})")
    plt.errorbar(ll_vec_best, vec_flux_obs, yerr=vec_fluxe_obs, fmt='o', color='#FCD12A',
                 label="Observed spectra with errors")
    plt.legend(loc="upper right")
    plt.xlabel("Central Wavelength (Å)")
    plt.ylabel("Flux (mJy)")
    plt.savefig(f"output_test/{i}_best_models.png", dpi=160)
    plt.close()

# ---- Plot the best BD and QSO models with the IGM spectrum and the source points with errors
    bdRA_f_best = (bdRA_data.columns[2 + int(BD_Chi2_min_ind)] / bdRA_l) * a_BD
    bdRA_fmJy_best = flux_from_cgs_to_mJy(bdRA_f_best, bdRA_l)
    plt.plot(bdRA_l, bdRA_fmJy_best, color="#dab894", label=f'Brown Dwarf spectrum with {BD_Chi2_min_temp}', zorder=0)
    plt.plot(w_rest, flux_mJY_best, color="#a7bed3", label=f'Quasar spectrum with z={QSO_Chi2_min_z}', zorder=0)
    plt.scatter(ll_vec_best, vec_flux_model_BD_best, color="#836853", label=f"Best BD template ({BD_Chi2_min_temp})", zorder=5)
    plt.scatter(ll_vec_best, vec_flux_model_QSO_best, color="#004987", label=f"Best QSO template (z={QSO_Chi2_min_z})", zorder=5)
    plt.errorbar(ll_vec_best, vec_flux_obs, yerr=vec_fluxe_obs, fmt='o', color='#FCD12A',label="Observed spectra with errors", zorder=10)
    plt.legend(loc="upper right")
    plt.xlabel("Central Wavelength (Å)")
    plt.ylabel("Flux (mJy)")
    plt.xlim(0, 55000)
    plt.savefig(f"output_test/{i}_BD_QSO_best_models.png", dpi=160)
    plt.close()


