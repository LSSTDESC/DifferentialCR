# Import packages and define functions

import numpy as np
import matplotlib.pyplot as plt
import galsim

from astropy.io.votable import parse
from scipy.interpolate import interp1d

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, Angle, AltAz, SkyCoord

# import gala.coordinates as gc
import astropy.coordinates as coord

from scipy.optimize import curve_fit

from scipy import integrate

import os

# Get the directory where THIS utils file lives
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))



def apply_filter(wvl, data, band = 'g', rm_leakage = True, cutoff = 0.002):
    '''
    wvl - a list of the wavelengths in nm
    data - a N x M list of SEDs, N = number of SEDs, M = len of wvl
    returns data with the band filter applied to each SEDwvl
    '''
    
    filter_file = os.path.join(_UTILS_DIR, f'filter_files/total_{band}.dat')
    filter_band = np.loadtxt(filter_file).T #filter_band[0] -> wavelengths, filter_band[1] -> filter pass fraction

    # remove filter leakage: set lower bound of filter to 0.001 throughput
    if rm_leakage:
        leakage_mask = filter_band[1] < cutoff
        filter_band[1][leakage_mask] = np.zeros(np.sum(leakage_mask))
    
    func = interp1d(filter_band[0], filter_band[1])
    SED_filter = func(wvl)
    
    return data * SED_filter



from astropy.modeling.models import BlackBody
from astropy.constants import c
import astropy.units as u


def BB(wave, temp, scale):
    # returns blackbody flux in units of erg/cm^2/s/A
    temp = temp * u.K
    wave = wave * u.nm
    scale = scale * u.erg / (u.cm**2 * u.s * u.AA * u.sr)

    bb = BlackBody(temperature = temp, scale = scale)

    return bb(wave).value


def apply_DCR(wave, filtered_data, zenith, pressure, temperature, H2O_pressure):
    '''
    wave - list of wavelength points
    filtered_data - dictionary, keys: bands, items: list of flux densities each SED as viewed in that band 

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


def mean_and_std(x_array, weights_dict, param_array = [None], multi = False):
    '''
    Returns the expectation (mean) and square root of the variance (std) of param_array: <param> and <(param - mean)^2>
    x_array : array to integrate over
    weights_dict : dictionary of distributions 
    param_array: parameter to find the mean and std of (default = x_array)
    multi : boolean, sets if there are multiple distributions in each dictionary entry to integrate

    '''
    
    if param_array[0] == None:
        param_array = x_array
    
    x_transpose = np.array(param_array).T #create a vertical array for purposes of matrix multiplication below

    means = {}
    stdevs = {}

    if multi:
        for band, weights in weights_dict.items():
            mean = np.trapezoid(x = x_array,y =  weights * x_transpose, axis = 1)/np.trapezoid(x = x_array, y = weights, axis =1)
            means[band] = mean
            stdevs[band] = np.sqrt(np.trapezoid(x = x_array, y = weights * (x_transpose - mean[:, np.newaxis])**2, axis=1) / np.trapezoid(x = x_array, y = weights, axis=1))

            # print(f'{band} (average): mean = {np.round(np.mean(mean), 4)}, stdev = {np.round(np.mean(stdevs[band]), 4)}')
    else:
        for band, weights in weights_dict.items():
            mean = np.trapezoid(x = x_array, y = weights * x_transpose)/np.trapezoid(x = x_array, y = weights)
            means[band] = mean
            stdevs[band] = np.sqrt(np.trapezoid(x = x_array, y = weights * (x_transpose - mean)**2) / np.trapezoid(x = x_array,y = weights))
        

    return means, stdevs
    
def weighted_avg_and_std(x_array, weights_dict, multi = False):

    """
    Don't use, deprecated and only works if x_array is uniformly sampled. mean_and_std() is much more reliable
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


def parallactic_angle_from_radec(ra, dec, mjd, lat = -30.244633 * u.deg, lon = -70.74941 * u.deg, height=2647*u.m * u.deg, sidereal_kind='apparent'):
    """
    Compute parallactic angle q (Angle) for given RA/Dec and observing site/time.

    Parameters
    ----------
    ra : astropy.units.Quantity or array-like (angle) -- e.g. 120*u.deg or [..]*u.deg
    dec: astropy.units.Quantity (angle)
    mjd: float or array-like (MJD)
    lat : astropy.units.Quantity (angle) -- observer latitude (positive north)
    lon : astropy.units.Quantity (angle) -- observer longitude (east positive; astropy accepts this)
    height: astropy.units.Quantity (length), optional, default 0*m
    sidereal_kind: 'apparent' or 'mean' (which sidereal time to use)

    Returns
    -------
    q : astropy.units.Quantity (angle) parallactic angle in radians (use .to(u.deg) to show degrees)
    """
    # ensure astropy Quantities
    ra = Angle(ra)
    dec = Angle(dec)
    lat = Angle(lat)
    lon = Angle(lon)

    # Time object
    t = Time(mjd, format='mjd', scale='utc')

    # Earth location
    loc = EarthLocation(lat=lat, lon=lon, height=height)

    # Local sidereal time at observer longitude (returns an Angle)
    lst = t.sidereal_time(kind=sidereal_kind, longitude=loc.lon)

    # Hour angle H = LST - RA; wrap to [-180, 180) to keep sines/cosines numerically stable
    H = (lst - ra).wrap_at(180*u.deg)

    # Convert to radians for numpy trig
    H_rad = H.to(u.rad).value
    phi = lat.to(u.rad).value
    delta = dec.to(u.rad).value

    # Compute using arctan2 to get correct quadrant
    numerator = np.sin(H_rad)
    denominator = np.tan(phi) * np.cos(delta) - np.sin(delta) * np.cos(H_rad)

    q_rad = np.arctan2(numerator, denominator)   # result in radians
    q = q_rad * u.rad

    # normalize to (-180,180] or whatever you prefer — here we return angle wrapped to [-180,180)
    q = Angle(q).wrap_at(360*u.deg)
    return q

def parallactic_angle_from_radec_fast(table, lat = -30.244633 * u.deg, lon = -70.74941 * u.deg, height=2647*u.m * u.deg, sidereal_kind='mean'):
    '''
    Parallel version of the above function that takes in a table with columns 'ra', 'dec', and 'expMidptMJD'
      and returns the table with the parallactic angle added as a new column 'q'
    '''

    loc = EarthLocation(lat = lat, lon = lon, height = height)

    times = Time(table['expMidptMJD'], format = 'mjd', scale = 'utc')
    lst = times.sidereal_time(kind = sidereal_kind, longitude = loc.lon)
    HA = (lst - Angle(list(table['ra'])* u.deg)).wrap_at(180*u.deg)


    H_rad = HA.to(u.rad).value
    phi = Angle(lat * u.deg).to(u.rad).value #lat * np.pi/180
    delta = Angle(np.array(table['dec'])*u.deg).to(u.rad).value #np.array(table['dec_1']) * np.pi/180

    numerator = np.sin(H_rad)
    denominator = np.tan(phi) * np.cos(delta) - np.sin(delta) * np.cos(H_rad)

    q_rad = np.arctan2(numerator, denominator)   # result in radians
    q = q_rad * u.rad

    # normalize to (-180,180] or whatever you prefer — here we return angle wrapped to [-180,180)
    q = Angle(q).wrap_at(360*u.deg)

    table['q'] = q.value

    return table

def add_zenith_coordinates(table, lat=-30.244633, lon=-70.749417, height=2647):
    """
    Adds zenith RA/Dec coordinates for each row in the table.
    Assumes columns 'expMidptMJD' for time.
    """
    location = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=height*u.m)
    zenith_ra = []
    zenith_dec = []
    for mjd in table['expMidptMJD']:
        obstime = Time(mjd, format='mjd', scale='utc')
        altaz = AltAz(alt=90*u.deg, az=0*u.deg, location=location, obstime=obstime)
        zenith = SkyCoord(altaz)
        zenith_icrs = zenith.transform_to('icrs')
        zenith_ra.append(zenith_icrs.ra.deg)
        zenith_dec.append(zenith_icrs.dec.deg)
    table['zenith_ra'] = zenith_ra
    table['zenith_dec'] = zenith_dec
    return table


def print_dp1_object_summary(object_table, ra, dec):
    """
    Print a formatted summary for the nearest entry in a DP1 Object query result.

    Parameters
    ----------
    object_table : astropy.table.Table
        Table returned from a dp1.Object query.
    ra, dec : float
        Query coordinates in degrees.
    """

    def _safe_fmt(value, ndp=4):
        try:
            numeric_value = float(value)
            if np.isfinite(numeric_value):
                return f"{numeric_value:.{ndp}f}"
        except Exception:
            pass
        return "nan"

    def _safe_color(mag_a, mag_b):
        try:
            value_a = float(mag_a)
            value_b = float(mag_b)
            if np.isfinite(value_a) and np.isfinite(value_b):
                return value_a - value_b
        except Exception:
            pass
        return np.nan

    if len(object_table) == 0:
        print("No objects found in dp1.Object for this RA/Dec and search radius.")
        return None

    delta_ra = (np.array(object_table['coord_ra']) - ra) * np.cos(np.deg2rad(dec))
    delta_dec = np.array(object_table['coord_dec']) - dec
    sep_arcsec = np.sqrt(delta_ra**2 + delta_dec**2) * 3600

    nearest_index = int(np.argmin(sep_arcsec))
    nearest_object = object_table[nearest_index]

    magnitudes = {
        'u': nearest_object['u_cModelMag'],
        'g': nearest_object['g_cModelMag'],
        'r': nearest_object['r_cModelMag'],
        'i': nearest_object['i_cModelMag'],
        'z': nearest_object['z_cModelMag'],
        'y': nearest_object['y_cModelMag'],
    }

    colors = {
        'u-g': _safe_color(magnitudes['u'], magnitudes['g']),
        'g-r': _safe_color(magnitudes['g'], magnitudes['r']),
        'r-i': _safe_color(magnitudes['r'], magnitudes['i']),
        'i-z': _safe_color(magnitudes['i'], magnitudes['z']),
        'z-y': _safe_color(magnitudes['z'], magnitudes['y']),
    }

    ref_extendedness = nearest_object['refExtendedness']
    size_extendedness = nearest_object['refSizeExtendedness']
    try:
        is_extended = bool(ref_extendedness > 0.5)
    except Exception:
        is_extended = False

    print("=" * 68)
    print("DP1 Object Summary (nearest match in search region)")
    print("=" * 68)
    print(f"Query position (deg):  RA={ra:.8f}, Dec={dec:.8f}")
    print(
        f"Matched position (deg): RA={_safe_fmt(nearest_object['coord_ra'], 8)}, Dec={_safe_fmt(nearest_object['coord_dec'], 8)}"
    )
    print(f"Separation:            {_safe_fmt(sep_arcsec[nearest_index], 3)} arcsec")
    print(f"Object ID:             {nearest_object['objectId']}")
    print(f"Tract/Patch:           {nearest_object['tract']} / {nearest_object['patch']}")
    print("-")
    print("Morphology")
    print(f"  Extended (refExtendedness > 0.5): {is_extended}")
    print(f"  refExtendedness:      {_safe_fmt(ref_extendedness, 4)}")
    print(f"  refSizeExtendedness:  {_safe_fmt(size_extendedness, 4)}")
    print(
        "  shape_xx, shape_xy, shape_yy: "
        f"{_safe_fmt(nearest_object['shape_xx'], 5)}, "
        f"{_safe_fmt(nearest_object['shape_xy'], 5)}, "
        f"{_safe_fmt(nearest_object['shape_yy'], 5)}"
    )
    print("-")
    print("cModel Magnitudes")
    print(
        "  u={u}, g={g}, r={r}, i={i}, z={z}, y={y}".format(
            u=_safe_fmt(magnitudes['u'], 4),
            g=_safe_fmt(magnitudes['g'], 4),
            r=_safe_fmt(magnitudes['r'], 4),
            i=_safe_fmt(magnitudes['i'], 4),
            z=_safe_fmt(magnitudes['z'], 4),
            y=_safe_fmt(magnitudes['y'], 4),
        )
    )
    print("-")
    print("Colors (cModel)")
    print(
        "  u-g={ug}, g-r={gr}, r-i={ri}, i-z={iz}, z-y={zy}".format(
            ug=_safe_fmt(colors['u-g'], 4),
            gr=_safe_fmt(colors['g-r'], 4),
            ri=_safe_fmt(colors['r-i'], 4),
            iz=_safe_fmt(colors['i-z'], 4),
            zy=_safe_fmt(colors['z-y'], 4),
        )
    )
    print("=" * 68)

    return nearest_object




def rebin(x, y, stepsize = None, newx_edges = None, median = True):

    if newx_edges is None:        
        newx_edges = np.arange(min(x), max(x) + stepsize, stepsize)
    
    newx = newx_edges[0:-1] + (newx_edges[1:] - newx_edges[0:-1])/2
        

    binned_data = np.zeros(len(newx_edges) - 1)    
    binned_err = np.zeros(len(newx_edges) - 1)

    bin_ind = np.digitize(x, newx_edges) - 1
    
    for i in range(len(binned_data)):
        mask = bin_ind == i
        if median:
            newx[i] = np.median(x[mask])
            binned_data[i] = np.median(y[mask])
        else:
            newx[i] = np.mean(x[mask])
            binned_data[i] = np.mean(y[mask])
        binned_err[i] = np.std(y[mask])/np.sqrt(np.sum(mask))

    return newx, binned_data, binned_err



def add_parallactic_angle(table, ra_col = 'ra', dec_col = 'dec', expMidptMJD_col = 'expMidptMJD', lat = -30.244633,lon = -70.749417, height = 2647):
    # Calculate the parallactic angle and add to the table
    
    loc = EarthLocation(lat = lat * u.deg, lon = lon * u.deg, height = height * u.m)
    
    times = Time(table[expMidptMJD_col], format = 'mjd', scale = 'utc')
    lst = times.sidereal_time(kind = 'mean', longitude = loc.lon)
    HA = (lst - Angle(list(table[ra_col])* u.deg)).wrap_at(180*u.deg)
    
    H_rad = HA.to(u.rad).value
    phi = Angle(lat * u.deg).to(u.rad).value #lat * np.pi/180
    delta = Angle(np.array(table[dec_col])*u.deg).to(u.rad).value #np.array(master['dec_1']) * np.pi/180
    
    numerator = np.sin(H_rad)
    denominator = np.tan(phi) * np.cos(delta) - np.sin(delta) * np.cos(H_rad)
    
    q_rad = np.arctan2(numerator, denominator)   # result in radians
    q = q_rad * u.rad
    
    # normalize to (-180,180] or whatever you prefer — here we return angle wrapped to [-180,180)
    q = Angle(q).wrap_at(360*u.deg)

    table['q'] = q.value

    return table


def synth_fnu(wl, Flam, band, wl_units = 'nm', clean_filters = True):
    '''
    Calculates a synthetic flux
    wl - wavelength array in apropriate units (see wl_units)
    Flam - SED (unfiltered) in in erg/cm^2/s/A
    wl_units - 'nm' or 'A'
    '''
    
    if wl_units == 'nm':
        wl_A = wl * 10
        wl_nm = wl
    elif wl_units == 'A':
        wl_A = wl
        wl_nm = wl/10

    if clean_filters == True:
        filter_file = f'filter_files/noleak_{band}.dat'
    else:
        filter_file = f'filter_files/total_{band}.dat'
        
    filter_band = np.loadtxt(filter_file).T #filter_band[0] -> wavelengths, filter_band[1] -> filter pass fraction

    T = interp1d(filter_band[0], filter_band[1])(wl_nm)

    num = np.trapezoid(Flam * T * wl_A, wl_A)
    den = np.trapezoid(T / wl_A, wl_A)
    
    return num / den   # proportional to <f_nu>


import matplotlib.patches as mpatches

def plot_ellipse(ixx, iyy, ixy, center=(0, 0), scale=1, ax=None, **kwargs):
    """
    Plot an ellipse defined by second moments ixx, iyy, ixy.

    The covariance matrix [[ixx, ixy], [ixy, iyy]] is diagonalized to get
    the semi-axes (sqrt of eigenvalues) and orientation angle.

    Parameters
    ----------
    ixx, iyy, ixy : float
        Second moments (same units, e.g. arcsec² or pix²).
    center : (float, float)
        (x, y) center of the ellipse.
    scale : float
        Multiply semi-axes by this factor (e.g. scale=2 for a 2-sigma ellipse).
    ax : matplotlib Axes, optional
        Axes to draw on; defaults to current axes.
    **kwargs
        Passed to matplotlib.patches.Ellipse (e.g. color, alpha, fill).
    """
    if ax is None:
        ax = plt.gca()

    cov = np.array([[ixx, ixy], [ixy, iyy]])
    eigvals, eigvecs = np.linalg.eigh(cov)          # eigenvalues in ascending order
    eigvals = np.maximum(eigvals, 0)                 # guard against tiny negatives
    semi_minor, semi_major = np.sqrt(eigvals) * scale
    # angle of the major axis (eigvec corresponding to largest eigenvalue)
    angle_rad = np.arctan2(eigvecs[1, 1], eigvecs[0, 1])
    angle_deg = np.degrees(angle_rad)

    kwargs.setdefault('fill', False)
    kwargs.setdefault('edgecolor', 'k')
    ellipse = mpatches.Ellipse(
        xy=center,
        width=2 * semi_major,
        height=2 * semi_minor,
        angle=angle_deg,
        **kwargs
    )
    ax.add_patch(ellipse)
    return ellipse


def foo():
    
    print('foo')

    