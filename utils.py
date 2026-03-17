# Import packages and define functions

import numpy as np
import matplotlib.pyplot as plt
import galsim

from astropy.io.votable import parse
from scipy.interpolate import interp1d

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, Angle, AltAz, SkyCoord

import gala.coordinates as gc
import astropy.coordinates as coord

from scipy.optimize import curve_fit

from scipy import integrate


def apply_filter(wvl, data, band = 'g', rm_leakage = True):
    '''
    wvl - a list of the wavelengths in angstroms
    data - a N x M list of SEDs, N = number of SEDs, M = len of wvl
    returns data with the band filter applied to each SED
    '''
    
    filter_file = f'filter_files/total_{band}.dat'
    filter_band = np.loadtxt(filter_file).T #filter_band[0] -> wavelengths, filter_band[1] -> filter pass fraction

    # remove filter leakage: set lower bound of filter to 0.001 throughput
    if rm_leakage:
        leakage_mask = filter_band[1] < 0.001
        filter_band[1][leakage_mask] = np.zeros(np.sum(leakage_mask))
    
    func = interp1d(filter_band[0], filter_band[1])
    SED_filter = func(wvl)
    
    return data * SED_filter


from astropy.modeling.models import BlackBody
from astropy.constants import c
import astropy.units as u


def BB(wave, temp, scale):
    temp = temp * u.K
    wave = wave * u.nm
    scale = scale * u.erg / (u.cm**2 * u.s * u.AA * u.sr)

    bb = BlackBody(temperature = temp, scale = scale)
    return bb(wave).value


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
    Parallel version of the above function that takes in a table with columns 'ra_1', 'dec_1', and 'expMidptMJD'
      and returns the table with the parallactic angle added as a new column 'q'
    '''

    loc = EarthLocation(lat = lat, lon = lon, height = height)

    times = Time(table['expMidptMJD'], format = 'mjd', scale = 'utc')
    lst = times.sidereal_time(kind = sidereal_kind, longitude = loc.lon)
    HA = (lst - Angle(list(table['ra_1'])* u.deg)).wrap_at(180*u.deg)


    H_rad = HA.to(u.rad).value
    phi = Angle(lat * u.deg).to(u.rad).value #lat * np.pi/180
    delta = Angle(np.array(table['dec_1'])*u.deg).to(u.rad).value #np.array(table['dec_1']) * np.pi/180

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


def foo():
    print('foo')