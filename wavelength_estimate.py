import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import galsim
import os
from astropy.table import Table

from scipy.interpolate import interp1d

colors = {'u' : 'purple', 'g':'blue', 'r': 'green', 'i': 'orange', 'z' : 'magenta', 'y' : 'red'}

from utils import apply_filter
# def apply_filter(wvl, data, band = 'g', rm_leakage = True):
#     '''
#     wvl - a list of the wavelengths in angstroms
#     data - a N x M list of SEDs, N = number of SEDs, M = len of wvl
#     returns data with the band filter applied to each SED
#     '''
    
#     filter_file = f'filter_files/total_{band}.dat'
#     filter_band = np.loadtxt(filter_file).T #filter_band[0] -> wavelengths, filter_band[1] -> filter pass fraction

#     # remove filter leakage: set lower bound of filter to 0.001 throughput
#     if rm_leakage:
#         leakage_mask = filter_band[1] < 0.001
#         filter_band[1][leakage_mask] = np.zeros(np.sum(leakage_mask))
    
#     func = interp1d(filter_band[0], filter_band[1])
#     SED_filter = func(wvl)
    
#     return data * SED_filter
    

#Pat and Josh's paper atmospheric conditions 
pressure=69.328
temperature=293.15
H2O_pressure=1.067


def apply_DCR(wave, filtered_data, zenith, pressure, temperature, H2O_pressure):
    '''
    wave - list of wavelength points
    filtered_data - dictionary, keys: bands, items: list of flux throughputs for each SED as viewed in that band 

    Takes in SED data with the filter throughputs already applied, returns refraction_angles (mapped from wave) and filtered_refracted_data (the flux density over refraction angles)

    '''
    dwvl = 0.1 #wavelength step
    wave_temp = np.arange(290, 1160, dwvl)

    #Get dR/dwavelength function
    refraction = galsim.dcr.get_refraction(
        wave_temp,
        zenith_angle=zenith * galsim.degrees,
        pressure=pressure,  # kPa
        temperature=temperature,  # K
        H2O_pressure=H2O_pressure,  # kPa
        ) * 180 * 3600 / np.pi #convert from radians to arcsec

    dR_dwvl = (refraction[1:] - refraction[0:-1])/dwvl
    wavelengths_midpoints = wave_temp[0:-1] + dwvl/2
    
    dRdwvl_func = interp1d(wavelengths_midpoints, dR_dwvl)

    #Get the refraction angle of each passed in wavelength 
    refraction_angles = galsim.dcr.get_refraction(
            wave,
            zenith_angle=zenith * galsim.degrees,
            pressure=pressure,  # kPa
            temperature=temperature,  # K
            H2O_pressure=H2O_pressure,  # kPa
            ) * 180 * 3600 / np.pi #convert from radians to arcsec
    
    
    filtered_refracted_data = {} #dictionary which will store dN/dR 
    
    for band, data in filtered_data.items():
        filtered_refracted_data[band] = np.abs(data / dRdwvl_func(wave))

    return refraction_angles, filtered_refracted_data

def weighted_avg_and_std(x_array, weights_dict, multi = False):

    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    x_transpose = np.array(x_array).T #create a vertical array for purposes of matrix multiplication below

    means = {}
    stdevs = {}

    if multi:
        for band, weights in weights_dict.items():
            mean = np.sum(weights * x_transpose, axis = 1)/np.sum(weights, axis =1)
            means[band] = mean
            stdevs[band] = np.sqrt(np.sum(weights * (x_transpose - mean[:, np.newaxis])**2, axis=1) / np.sum(weights, axis=1))

            # print(f'{band} (average): mean = {np.round(np.mean(mean), 4)}, stdev = {np.round(np.mean(stdevs[band]), 4)}')
    else:
        for band, weights in weights_dict.items():
            mean = np.sum(weights * x_transpose)/np.sum(weights)
            means[band] = mean
            stdevs[band] = np.sqrt(np.sum(weights * (x_transpose - mean)**2) / np.sum(weights))
        

    return means, stdevs



from astropy.modeling.models import BlackBody
from astropy.constants import c
import astropy.units as u


def BB(wave, temp, scale):
    temp = temp * u.K
    wave = wave * u.AA
    scale = scale * u.erg / (u.cm**2 * u.s * u.AA * u.sr)

    bb = BlackBody(temperature = temp, scale = scale)
    return bb(wave).value



def generate_templates(reference_temperature = 4500, sigma = 100, gaussian = False, raw_wavelengths = False):

    # generate a bunch of template SEDs with well defined mean wavelengths (gaussian profiles)

    wavelengths = np.linspace(300, 1100, 5000) #angstroms
    # sigma = 100

    if gaussian:
        # --- Gaussian SEDs
        mus = np.linspace(100, 1300, 5000)
        SEDs = np.array([np.exp(-0.5 * ((wavelengths - mu) / sigma)**2) for mu in mus])

    else:
        # --- power law fluxes
        alphas = np.linspace(-20, 20, 5000)
        SEDs = np.array([(wavelengths/10)**(alpha) for alpha in alphas])

    filtered_SEDs = {}
    for band in 'ugrizy':
        filtered_SEDs[band] = apply_filter(wavelengths, SEDs , band = band)


    if raw_wavelengths:
        band_edges = {'u': [334., 400.], 'g': [395., 560.], 'r': [542., 699.], 'i': [680., 831.], 'z' :[807., 931.], 'y' :[ 916., 1054.]}
        edgecut_SEDs = {}
        
        for band in 'ugrizy':
                    band_mask = (wavelengths > band_edges[band][0]) * (wavelengths < band_edges[band][1])
                    edgecut_SEDs[band] = SEDs[:]
                    edgecut_SEDs[band][~band_mask] = 0

        mean_wavelength, std_wavelength = weighted_avg_and_std(wavelengths, edgecut_SEDs, multi = True)
            
    else:
        mean_wavelength, std_wavelength = weighted_avg_and_std(wavelengths, filtered_SEDs, multi = True)

    # plt.plot(wavelengths, mean_wavelength['u'], label = 'g mean wavelength', color = 'blue')
    # plt.ylabel('Weighted Mean Wavelength (Angstroms)')
    # plt.xlabel('Input SED Mean Wavelength (Angstroms)')

    R, DCR = apply_DCR(
        wavelengths,
        filtered_SEDs,
        zenith=45,  # degrees
        pressure=pressure,  # kPa
        temperature=temperature,  # K
        H2O_pressure=H2O_pressure,  # kPa
        )

    mean_R0, std_R0 = weighted_avg_and_std(R, DCR, multi = True)


    # get the reference wavelengths for the calibration star

    reference_temperature = 4500  # K

    bb_flux = BB(wavelengths * 10, reference_temperature, 1e-8)

    # plt.plot(wavelengths, bb_flux, color = 'k', label = 'Reference SED')

    ref_filtered_flux = {}
    for i, band in enumerate('ugrizy'):
        ref_filtered_flux[band] = apply_filter(wavelengths, bb_flux, band = band)

    ref_wave, ref_std = weighted_avg_and_std(wavelengths, ref_filtered_flux, multi = False)
    # plt.show()
    # plt.close()

    R_, ref_DCR = apply_DCR(
        wavelengths,
        ref_filtered_flux,
        zenith=45,  # degrees
        pressure=pressure,  # kPa
        temperature=temperature,  # K
        H2O_pressure=H2O_pressure,  # kPa
        )

    ref_R0, ref_std_R0 = weighted_avg_and_std(R, ref_DCR, multi = False)

    return mean_wavelength, mean_R0, ref_R0

    # print(ref_R0)

mean_wavelength, mean_R0, ref_R0 = generate_templates()

def estimate_mean_wave(delta_R0, band, mean_wavelength = mean_wavelength, mean_R0 = mean_R0, ref_R0 = ref_R0):
    '''
    delta_R0 - difference in refraction angle from reference star in arcsec
    band - 'ugrizy'
    
    returns estimated mean wavelength in angstroms
    '''

    func = interp1d(mean_R0[band] - ref_R0[band], mean_wavelength[band], bounds_error = False, fill_value = np.nan)

    return func(delta_R0)



from galsim.dcr import get_refraction, air_refractive_index_minus_one

band_edges = {'u': (np.float64(333.6336336336336), np.float64(400.1001001001001)), 
              'g': (np.float64(395.2952952952953), np.float64(560.2602602602603)), 
              'r': (np.float64(541.8418418418419), np.float64(698.7987987987988)), 
              'i': (np.float64(679.5795795795796), np.float64(830.9309309309309)), 
              'z': (np.float64(806.9069069069069), np.float64(931.031031031031)), 
              'y': (np.float64(915.8158158158159), np.float64(1054.3543543543544))}

band_wavelengths = {}
R_analytic = {}
for band in 'ugrizy':
    band_wavelengths[band] = np.linspace(band_edges[band][0], band_edges[band][1], 1000)

    R45 = get_refraction(
        band_wavelengths[band],
        zenith_angle=45 * galsim.degrees,
        pressure=pressure,  # kPa
        temperature=temperature,  # K
        H2O_pressure=H2O_pressure,  # kPa
        ) * 180 * 3600 / np.pi #convert from radians to arcsec
    R_analytic[band] = R45


def analytic_mean_wave(delta_R0, band, mean_wavelength = mean_wavelength, mean_R0 = mean_R0, ref_R0 = ref_R0):
    func = interp1d(R_analytic[band] - ref_R0[band], band_wavelengths[band], bounds_error = False, fill_value = np.nan)
    return func(delta_R0)

# for band in 'ugrizy':    
#     plt.plot(mean_wavelength[band], (mean_R0[band] - ref_R0[band])*1000, label = f'{band} band', color = colors[band])
#     print(min((mean_R0[band] - ref_R0[band])*1000), max((mean_R0[band] - ref_R0[band])*1000))


# plt.xlabel('Mean Wavelength (Angstroms)')
# plt.ylabel(r'$\Delta R_0$ (arcsec)')
