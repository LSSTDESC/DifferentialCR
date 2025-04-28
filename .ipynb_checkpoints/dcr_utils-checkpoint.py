import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import sqlite3
import astropy.units as u
import galsim
import os
import matplotlib.lines as mlines
from astropy.coordinates import SkyCoord
import astropy.units as u

#Must use rubin_sim conda environment.

#calculates when a filter throughput drops below 10 percent of its maximum value. 
#this determines the "width of the filter" used in the calculation of angle of refraction
#wavelengths is a 1D numpy array that encompass the range of wavelengths for a filter.
def calculate_fwhm(wavelengths, throughput, cutoff = 8005):
    max_index = np.argmax(throughput)
    half_max = throughput[max_index] / 10 #may change to actual half maximum. This will calculate the range of the filter
    
    # left index where throughput drops below half max
    left_idx = max_index
    for i in range(max_index, 0, -1):
        if throughput[i] < half_max:
            left_idx = i
            break
    
    # right index where throughput drops below half max
    right_idx = 0
    j = cutoff #throughput becomes zeros past the value of cutoff = 8005
    while throughput[j] <= half_max:
            right_idx = j
            j -= 1
            if throughput[j] > half_max:
                break
    
    fwhm = wavelengths[right_idx] - wavelengths[left_idx]
    return fwhm, wavelengths[left_idx], wavelengths[right_idx], left_idx, right_idx

#this function find the neares index for a given arbitrary value. Used in other functions in dcr_utils
def find_nearest_index(array, value):
    return (np.abs(array - value)).argmin()


#this function reads a filter file from the specified directory and returns its data.
#the filter file is expected to contain wavelength and throughput values.
#the function returns a NumPy array where each row corresponds to a data point.
def read_filter_data_from_directory(directory, filter_file):
    file_path = os.path.join(directory, filter_file)
    data = np.loadtxt(file_path)
    return data

#used make a plot of how the Trace and Area are scaled due to dcr. 
#(filter_files is a list of names of each file that contains the filter throughputs)
#directory is the directory that contains the filter files
#airmasses is an arbitrary array of airmass values
def calculate_refraction_and_seeing_effects(filter_files, directory, airmasses, calc_type = None, airmass_seeing = True, 
                                            wavelength_seeing = True, dcr = True, cutoff = 8005):
    wavelengths = np.linspace(300,1100, len(airmasses))
    # seeing effects:
    if airmass_seeing == True:
        airmass_seeing = airmasses ** (0.6) #added seeing effect (will do first since seeing + dcr isn't commutative)
    elif airmass_seeing == False:
        airmass_seeing = airmasses ** 0
    if wavelength_seeing == True:
        wavelength_seeing_exponent = 1.5/5
    else:
        wavelength_seeing_exponent = 0
    
    #initializing lists
    filter_refractions = []
    Area_Ratio = []
    Area_PSF = []
    Trace_PSF = []
    Trace_Ratio = []
    psf_width = []
    g_shear = []
    #change depending on value you'd want. Here, we assume PSF FWHM = 0.7 arcsec at 1 = 750 and airmass = 1
    nominal_psf_width = 0.70

    for i, filter_file in enumerate(filter_files):
        data = read_filter_data_from_directory(directory, filter_file)
        # this assumes that the values become zero past an index
        wavelength = data[:(cutoff + 1), 0]
        throughput = data[:(cutoff + 1), 1]
        fwhm, left_wavelength, right_wavelength, left_idx, right_idx = calculate_fwhm(wavelength, throughput)
        
        nominal_wavelength = find_average_wavelength(filter_files[3], wavelengths, directory, cutoff = 8005)
        left_idx_new = find_nearest_index(wavelengths, left_wavelength)
        right_idx_new = find_nearest_index(wavelengths, right_wavelength)
        avg_wavelength = (left_wavelength + right_wavelength) / 2
        avg_idx_new = find_nearest_index(wavelengths, avg_wavelength)
        psf_width.append(nominal_psf_width * airmass_seeing * (nominal_wavelength / wavelengths[avg_idx_new]) ** (wavelength_seeing_exponent))

        filter_refractions = calculate_filter_refractions(filter_files, directory, dcr = dcr, airmasses = airmasses, cutoff = 8005)
        
        Area_Ratio.append(2.35 * np.sqrt((psf_width[i] / 2.35) ** 2 + filter_refractions[i] ** 2 / 12) / psf_width[i])
        Area_PSF.append(np.pi * (psf_width[i] / 2.35) * np.sqrt((psf_width[i] / 2.35) ** 2 + filter_refractions[i] ** 2 / 12)) 
        Trace_Ratio.append((psf_width[i] ** 2 + (2.35 * filter_refractions[i]) ** 2 / 24) / psf_width[i] ** 2)
        Trace_PSF.append(2 * (psf_width[i]/2.35) ** 2 + filter_refractions[i] ** 2 / 12)
        g_shear.append((filter_refractions[i]) ** 2 /
                       (12 * (2 * (psf_width[i] / 2.35) ** 2 + filter_refractions[i] ** 2 / 12 + 2 * psf_width[i] / 
                        2.35 * np.sqrt((psf_width[i] / 2.35) ** 2 + filter_refractions[i] ** 2 / 12))))

    if calc_type == 'Area_PSF':
        return Area_PSF
    elif calc_type == 'Area_Ratio':
        return Area_Ratio
    elif calc_type == 'Trace_PSF':
        return Trace_PSF
    elif calc_type == 'Trace_Ratio':
        return Trace_Ratio
    elif calc_type == 'g_shear':
        return g_shear
    elif calc_type == None:
        return Area_PSF, Area_Ratio, Trace_PSF, Trace_Ratio, g_shear
    else:
        raise ValueError("Invalid calc_type specified. Choose from 'Area_PSF', 'Area_Ratio', 'Trace_PSF', 'Trace_Ratio', 'g_shear'.")

#find the difference in angle of refraction from one end of filter to other end. (Uses calculate_fwhm function)
#(filter_files is a list of names of each file that contains the filter throughputs)
#directory is the directory that contains the filter files
#airmasses is an arbitrary array of airmass values
def calculate_filter_refractions(filter_files, directory, airmasses, dcr = True, cutoff = 8005):
    wavelengths = np.linspace(300, 1100, len(airmasses))
    zeniths = []
    for airmass in airmasses:
        if airmass > 0:
            zeniths.append(np.arccos(1 / airmass))
        else:
            zeniths.append(np.nan)  # handle zero or negative airmass values
    zeniths = np.degrees(zeniths) * u.deg

    gs_refractions = np.zeros((len(zeniths), len(wavelengths)))
    for i, zenith in enumerate(zeniths):
        gs_refraction = 3600 * np.rad2deg(galsim.dcr.get_refraction(
            wavelengths,
            zenith_angle=zenith.to_value(u.deg) * galsim.degrees,
            pressure=70.0,  # kPa
            temperature=293.15,  # K
            H2O_pressure=0.0,  # kPa
        ))
        gs_refractions[i, :] = gs_refraction

    filter_refractions = []

    for i, filter_file in enumerate(filter_files):
        data = read_filter_data_from_directory(directory, filter_file)
        wavelength = data[:(cutoff + 1), 0]
        throughput = data[:(cutoff + 1), 1]
        fwhm, left_wavelength, right_wavelength, left_idx, right_idx = calculate_fwhm(wavelength, throughput)
        
        left_idx_new = find_nearest_index(wavelengths, left_wavelength)
        right_idx_new = find_nearest_index(wavelengths, right_wavelength)

        refraction_left = gs_refractions[:, left_idx_new]
        refraction_right = gs_refractions[:, right_idx_new]
        differential_refractions = abs(refraction_left - refraction_right) if dcr else np.zeros_like(airmasses)
        filter_refractions.append(differential_refractions)
    return filter_refractions

#calculates a quantile. uses cursor to extract specific visits from OpSim database (airmass.ipnyb notebook shows how to extract data from notebook).
#thank you Theo for showing me how to!
def calculate_quartile(db_fname, fraction = 0.5, filter = None):
    # open a connection to the database file
    con = sqlite3.connect(db_fname)
    cur = con.cursor()

    # query to get RA, Dec, and airmass of all observations
    if filter == None:
        res = cur.execute(f"""
            SELECT
                fieldRA,
                fieldDec,
                airmass
            FROM
                observations
            WHERE
                target = ''
                and not (note = 'twilight_near_sun, 0'
                or note = 'twilight_near_sun, 1'
                or note = 'twilight_near_sun, 2'
                or note = 'twilight_near_sun, 3')
        """)
    else:
        res = cur.execute(f"""
            SELECT
                fieldRA,
                fieldDec,
                airmass
            FROM
                observations
            WHERE
                target = '' and filter = '{filter}'
                and not (note = 'twilight_near_sun, 0'
                or note = 'twilight_near_sun, 1'
                or note = 'twilight_near_sun, 2'
                or note = 'twilight_near_sun, 3')
        """)
    data = np.array(res.fetchall())
    #START

    # obtain data in terms of an array with the same shape size(ra) = size(dec) = size(airmass)
    ra = data[:, 0]
    dec = data[:, 1]
    airmass = data[:, 2]

    #making the resolution of the map, as well as the pixels in the form of an array
    nside = 32
    sentinel_value = 0
    base_map = sentinel_value * np.ones(hp.nside2npix(nside))

    #this part is talking about the actual pixel size
    pix = hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)

    # so this creates an array of indeces that organizes the array(pix) in terms of their pixel number, 
    # and then also as a tie-breaker sorts the pixels in terms of their airmass
    sort_idx = np.lexsort((airmass, pix))

    # now we take all the different unique pixel values (i.e. total number of pixels)
    # and obtain the u_idx_sorted which gives the first index that that pixel is referenced
    # then counts_in_pix which states the number of visits was within that pixel/ times that pixel was mentioned
    unique_pix, u_idx_sorted, counts_in_pix = np.unique(pix[sort_idx], return_index=True, return_counts=True)

    #this defines what quartile in the airmass distribution of that pixel we want to be looking at

    #bin_centers is near the index that is at the value we want to see (i.e. at that specific airmass value)
    bin_centers = u_idx_sorted + (counts_in_pix - 1) * (fraction) #Change depending on the value you're looking at

    #we round up and down the bin_centers obtain the actual index to obtain the range of the sorted airmass values
    mids_low = airmass[sort_idx][np.floor(bin_centers).astype(int)]
    mids_hi = airmass[sort_idx][np.ceil(bin_centers).astype(int)]
    #average out the airmass value at those specific indeces
    medians = (mids_low + mids_hi) * (1/2)

    # map quartiles
    median_gi_map = base_map.copy() #array of the number of pixels
    median_gi_map[unique_pix] = medians
    return median_gi_map

#does the same as quartile, but find the average
def calculate_average_airmass(db_fname, filter = None):
    # open a connection to the database file
    con = sqlite3.connect(db_fname)
    cur = con.cursor()

    # Query to get RA, Dec, and airmass of all observations
    if filter == None:
        res = cur.execute(f"""
            SELECT
                fieldRA,
                fieldDec,
                airmass
            FROM
                observations
            WHERE
                target = ''
                and not (note = 'twilight_near_sun, 0'
                or note = 'twilight_near_sun, 1'
                or note = 'twilight_near_sun, 2'
                or note = 'twilight_near_sun, 3')
        """)
    else:
        res = cur.execute(f"""
            SELECT
                fieldRA,
                fieldDec,
                airmass
            FROM
                observations
            WHERE
                target = '' and filter = '{filter}'
                and not (note = 'twilight_near_sun, 0'
                or note = 'twilight_near_sun, 1'
                or note = 'twilight_near_sun, 2'
                or note = 'twilight_near_sun, 3')
        """)
    data = np.array(res.fetchall())
    #START
    # obtain data in terms of an array with the same shape size(ra) = size(dec) = size(airmass)
    ra = data[:, 0]
    dec = data[:, 1]
    airmass = data[:, 2]

    #making the resolution of the map, as well as the pixels in the form of an array
    nside = 32
    sentinel_value = 0
    base_map = sentinel_value * np.ones(hp.nside2npix(nside))

    #this part is talking about the actual pixel size
    pix = hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)
    # --- MEAN ---
    # Group them so that you can access all the objects assigned to each unique HEALPixel
    # `unique_pix` indexes the map pixels that contain at least one galaxy
    # `u_idx` gives you the index of the first occurrence of a unique value in `pix`
    # `idx_rep` gives you the indices into `unique_pix` that would return the original `pix` array
    # check the examples on the numpy docs for clarification
    unique_pix, u_idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
    # The mean g-i color per pixel is just the sum of that pixel's objects' g-i values divided by the number of objects in the pixel
    mean_gi_map = base_map.copy()
    mean_gi_map[unique_pix] = np.bincount(idx_rep, weights=airmass) / np.bincount(idx_rep)
    return mean_gi_map

#find the average wavelength for a particular filter
#this is used to find the middle of the "i-filter" and other filters
def find_average_wavelength(filter_file, wavelengths, directory, cutoff = 8005):
    wavelengths = np.linspace(300, 1100, len(wavelengths))

    data = read_filter_data_from_directory(directory, filter_file)
    wavelength = data[:(cutoff + 1), 0]
    throughput = data[:(cutoff + 1), 1]
    fwhm, left_wavelength, right_wavelength, left_idx, right_idx = calculate_fwhm(wavelength, throughput)

    avg_wavelength = (left_wavelength + right_wavelength) / 2
    avg_idx_new = find_nearest_index(wavelengths, avg_wavelength)
    return wavelengths[avg_idx_new]

#applies cosmology cuts by using:
# if metric_values[i] != '--': if value is not at that visit.
def apply_cosmology_cuts():
    ##############################################APPLY COSMOLOGY CUTS##########################################################
    # get the baseline db name
    opsim_fname = get_baseline()

    # set up the constraint, slicer, metric, and bundle - and then run the bundle
    nside = 128

    constraint = "scheduler_note not like '%DD%'"
    slicer = maf.slicers.HealpixSlicer(nside=nside)
    metric = maf.metrics.ExgalM5WithCuts(lsst_filter="i", extinction_cut=0.2,
                                        depth_cut=25.9, n_filters=6,
                                        )

    bundle_exgal = maf.MetricBundle(metric, slicer, constraint)         # we'll get the lsst_filter-band depth alongside the mask
    bundle_grp = maf.MetricBundleGroup([bundle_exgal], opsim_fname, out_dir=outdir)
    bundle_grp.run_all()
    return bundle_exgal.metric_values

#to import dcr utils, paste this without comment @ the top of python file:
"""
import sys
import os

module_path = '/Users/matthewlugatiman/Desktop/Galsim_Work/Week 9 Stuff/dcr_utils'
sys.path.append(module_path)

import dcr_utils as dcr
"""