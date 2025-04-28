from rubin_sim import maf
from rubin_sim.data import get_data_dir

try:
    from rubin_sim.data import get_baseline
except ImportError:
    from rubin_scheduler.data import get_baseline
import rubin_sim
from rubin_sim import data
from rubin_sim.data import get_baseline
import healpy as hp
import numpy as np
import sqlite3
from astropy.coordinates import SkyCoord
import astropy.units as u
import datashader as ds
import matplotlib.pyplot as plt
import matplotlib as mpl
from datashader.mpl_ext import dsshow
import pandas as pd
import galsim
import matplotlib.lines as mlines

# set up the outdir
outdir = 'test'
import os
os.makedirs(outdir, exist_ok=True)

def find_average_wavelength(filter_file, wavelengths, directory, cutoff = 8005):
    '''
    Returns the wavelength closest to the middle of a filter for a given filter 
    
    filter_file: txt file with two columns for the wavelength and throughput of each filter
    wavelengths: array of wavelengths from which you return the value closest to the middle of the chosen filter

    '''
    wavelengths = np.linspace(300, 1100, len(wavelengths)) 
    #it looks like this makes pretty much the same thing as the wavelengths object passed in?

    data = read_filter_data_from_directory(directory, filter_file)
    wavelength = data[:(cutoff + 1), 0] #remove some stuff at the high end?
    throughput = data[:(cutoff + 1), 1]
    fwhm, left_wavelength, right_wavelength, left_idx, right_idx = calculate_fwhm(wavelength, throughput)

    avg_wavelength = (left_wavelength + right_wavelength) / 2
    avg_idx_new = find_nearest_index(wavelengths, avg_wavelength)
    return wavelengths[avg_idx_new]



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
        #Seems like this information about the filters should just be saved once and referred back to
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

def find_nearest_index(array, value):
    return (np.abs(array - value)).argmin()

def calculate_fwhm(wavelength, throughput, cutoff = 8005):
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
    
    fwhm = wavelength[right_idx] - wavelength[left_idx]
    return fwhm, wavelength[left_idx], wavelength[right_idx], left_idx, right_idx

def read_filter_data_from_directory(directory, filter_file):
    file_path = os.path.join(directory, filter_file)
    data = np.loadtxt(file_path)
    return data

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


#calculates a quantile. uses cursor to extract specific visits from OpSim database (airmass.ipnyb notebook shows how to extract data from notebook).
#thank you Theo for showing me how to!
def calculate_quartile(opsim_fname, fraction = 0.5, filter = None):
    '''
    Returns a map of airmass values at the specified fraction quantile 
    '''
    # open a connection to the database file
    con = sqlite3.connect(opsim_fname)
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
                target_name = ''
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
                target_name = '' and filter = '{filter}'
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

    ############################ TO CONVERT HEALPIX BACK TO INDIVIDUAL VISITS ############################
    nside_convert = 128
    zenith_angle = am2deg(airmass)
    theta = np.radians(90.0 - dec)  # colatitude
    phi = np.radians(ra)            # longitude
    visit = hp.ang2pix(nside_convert, theta, phi)

    print(len(ra))
    ra_new, dec_new, airmass_new = [], [], []
    metric_values = apply_cosmology_cuts()
    # Iterate through the metric values and filter the data
    for visit_index, pixel in enumerate(visit):  # visit contains pixel indices
        if metric_values[pixel] != '--':
            # Append corresponding data
            ra_new.append(ra[visit_index])
            dec_new.append(dec[visit_index])
            airmass_new.append(airmass[visit_index])

    # Convert lists to NumPy arrays
    ra_new = np.array(ra_new)
    dec_new = np.array(dec_new)
    airmass_new = np.array(airmass_new)

    #######################################################################################################

    #making the resolution of the map, as well as the pixels in the form of an array
    nside = 32
    sentinel_value = 0
    base_map = sentinel_value * np.ones(hp.nside2npix(nside))

    #this part is talking about the actual pixel size
    pix = hp.ang2pix(nside, ra_new, dec_new, lonlat=True, nest=True)

    # so this creates an array of indeces that organizes the array(pix) in terms of their pixel number, 
    # and then also as a tie-breaker sorts the pixels in terms of their airmass
    sort_idx = np.lexsort((airmass_new, pix))

    # now we take all the different unique pixel values (i.e. total number of pixels)
    # and obtain the u_idx_sorted which gives the first index that that pixel is referenced
    # then counts_in_pix which states the number of visits was within that pixel/ times that pixel was mentioned
    unique_pix, u_idx_sorted, counts_in_pix = np.unique(pix[sort_idx], return_index=True, return_counts=True)

    #this defines what quartile in the airmass distribution of that pixel we want to be looking at

    #bin_centers is near the index that is at the value we want to see (i.e. at that specific airmass value)
    bin_centers = u_idx_sorted + (counts_in_pix - 1) * (fraction) #Change depending on the value you're looking at

    #we round up and down the bin_centers obtain the actual index to obtain the range of the sorted airmass values
    mids_low = airmass_new[sort_idx][np.floor(bin_centers).astype(int)]
    mids_hi = airmass_new[sort_idx][np.ceil(bin_centers).astype(int)]
    #average out the airmass value at those specific indeces
    medians = (mids_low + mids_hi) * (1/2)

    # map quartiles
    median_gi_map = base_map.copy() #array of the number of pixels
    median_gi_map[unique_pix] = medians
    return median_gi_map

def deg2am(deg):
    return 1 / np.cos(np.radians(deg))

def am2deg(am):
    am_clipped = np.clip(am, 1, None)
    return np.degrees(np.arccos(1 / am_clipped))


#######################################MAKE ELLIPTICITY PLOTS##############################################################
# Create a figure for the subplots (5 rows, 3 columns)
fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(12, 10), constrained_layout=False) #12, 10 may be the best
opsim_fname = get_baseline() #Look at this
directory = '/Documents/DESC/matthew_resources/filter_files' #MUST CHANGE
filter_files = ['total_u.dat', 'total_g.dat', 'total_r.dat', 'total_i.dat', 'total_z.dat', 'total_y.dat']
filters = ['u', 'g', 'r', 'i']
quartiles = [.50, .75, .95]

for i, filter in enumerate(filters):
    # Dictionary to map filter letters to their index
    filter_map = {'u': 0, 'g': 1, 'r': 2, 'i': 3}
    filter_file_index = filter_map.get(filter, 0) #return the index of the filter if in filter_map, otherwise return 0 --> u?
    
    for j, nth_quartile in enumerate(quartiles):
        # Calculate median_gi_map and g_shear for the given quartile and filter
        median_gi_map = calculate_quartile(opsim_fname, fraction=quartiles[j], filter=filter)
        g_shear = calculate_refraction_and_seeing_effects(filter_files, directory, 
                                                              airmass_seeing=True, wavelength_seeing=True, 
                                                              dcr=True, airmasses=median_gi_map, calc_type='g_shear')
        
        # Select the correct subplot
        ax = axs[i, j]
        plt.sca(ax)  # Set current axis
        
        # Plot the HEALPix map using mollview
        hp.mollview(g_shear[filter_file_index], title=f"Filter: {filter}, {nth_quartile}", 
                    cmap='magma', cbar=False, nest=True, min=0, max=0.16, hold=True)
        
        # Add graticules
        hp.graticule()
        print('-')
# Remove the last row's axes (7th row)
for j in range(3):
    fig.delaxes(axs[4, j])

vmin = 0
vmax = 0.16
# Create a ScalarMappable to normalize the color scale and create a global color bar
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
sm = mpl.cm.ScalarMappable(norm=norm, cmap='magma')
sm.set_array([])  # Required for ScalarMappable

# Add the color bar to the figure
cbar = fig.colorbar(sm, ax=axs, orientation='horizontal', pad = 0.2)

# Set the color bar label
cbar.set_label('Ellipticity')

plt.subplots_adjust(bottom=0.15)

# Save and show the figure
plt.show()
############################################################################################################