import numpy as np
import astropy.units as u
from lsst.geom import LinearTransform

def parallactic_angle_from_radec(ra, dec, mjd, lat = -30.244633 * u.deg, lon = -70.74941 * u.deg, height=2647 * u.m, sidereal_kind='mean'):
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
    from astropy.coordinates import EarthLocation, Angle, AltAz, SkyCoord
    from astropy.time import Time
    
    # ensure astropy Quantities
    ra = Angle(ra)
    dec = Angle(dec)
    lat = Angle(lat)
    lon = Angle(lon)

    # Time object
    t = Time(mjd, format='mjd', scale='utc')

    # Earth location
    loc = EarthLocation(lat=lat, lon=lon, height=height)
    print('loc', loc)

    # Local sidereal time at observer longitude (returns an Angle)
    lst = t.sidereal_time(kind=sidereal_kind, longitude=loc.lon)

    # Hour angle H = LST - RA; wrap to [-180, 180) to keep sines/cosines numerically stable
    H = (lst - ra).wrap_at(180*u.deg)

    # Convert to radians for numpy trig
    H_rad = H.to(u.rad).value
    phi = lat.to(u.rad).value
    delta = dec.to(u.rad).value
    print('H_rad', H_rad)
    print('phi', phi)
    print('delta', delta)

    # Compute using arctan2 to get correct quadrant
    numerator = np.sin(H_rad)
    denominator = np.tan(phi) * np.cos(delta) - np.sin(delta) * np.cos(H_rad)

    q_rad = np.arctan2(numerator, denominator)   # result in radians
    q = q_rad * u.rad

    # normalize to (-180,180] or whatever you prefer — here we return angle wrapped to [-180,180)
    q = Angle(q).wrap_at(360*u.deg)

    
    return q

def rotateXYtoAA(Ixx, Iyy, Ixy, rotTelPos):
    """
    Rotates moments in the x and y camera/telescope coord system to alt and az 

    Parameters
    ----------
    Ixx, Iyy, Ixy : lists of shape moments in the camera x and y coordinates

    rotTelPos : list of Telescope rotation angles in radians corresponding to each measurement

    Returns
    -------
    aaIxx, aaIyy, aaIxy : The moments in the new coordinate system [[I_alt_alt, I_alt_az],[I_alt_az, I_az_az]]
    """
    from lsst.afw.geom import Quadrupole
    from lsst.geom import LinearTransform
    crtp, srtp = np.cos(rotTelPos), np.sin(rotTelPos)
    aaRot = [np.array([[crtp[i], srtp[i]], [-srtp[i], crtp[i]]]) @ np.array([[0, 1], [1, 0]]) @ np.array([[-1, 0], [0, 1]]) for i in range(len(crtp))]

    aaIxx = []
    aaIyy = []
    aaIxy = []
    
    for i, Ixx_ in enumerate(Ixx):
        shape = Quadrupole(Ixx_, Iyy[i], Ixy[i])
        rotShape = shape.transform(LinearTransform(aaRot[i]))    
        
        aaIxx.append(rotShape.getIxx()) 
        aaIyy.append(rotShape.getIyy()) 
        aaIxy.append(rotShape.getIxy())


    return np.array(aaIxx), np.array(aaIyy), np.array(aaIxy)


def rotateXYtoNW(Ixx, Iyy, Ixy, rotSkyPos):
    """
    Rotates moments in the x and y camera/telescope coord system to ra (w) and dec (n)

    Parameters
    ----------
    Ixx, Iyy, Ixy : lists of shape moments in the camera x and y coordinates

    rotSkyPos : list of Sky rotation angles in radians corresponding to each observation

    Returns
    -------
    nwIxx, nwIyy, nwIxy : The moments in the new coordinate system [[I_ra_ra, I_ra_dec],[I_ra_dec, I_dec_dec]]
    """
    from lsst.afw.geom import Quadrupole
    from lsst.geom import LinearTransform
    
    crsp, srsp = np.cos(rotSkyPos), np.sin(rotSkyPos)
    nwRot = [np.array([[crsp[i], -srsp[i]], [srsp[i], crsp[i]]]) for i in range(len(crsp))]

    nwIxx = []
    nwIyy = []
    nwIxy = []

    
    for i, Ixx_ in enumerate(Ixx):
        shape = Quadrupole(Ixx_, Iyy[i], Ixy[i])
        rotShape = shape.transform(LinearTransform(nwRot[i]))
    
        nwIxx.append(rotShape.getIxx()) 
        nwIyy.append(rotShape.getIyy()) 
        nwIxy.append(rotShape.getIxy())


    return np.array(nwIxx), np.array(nwIyy), np.array(nwIxy)

def rotateNWtoAA(nwIxx, nwIyy, nwIxy, q):
    """
    Rotates moments in the ra (w) and dec (n) to alt and az
    
    Parameters
    ----------
    nwIxx, nwIyy, nwIxy : lists of shape moments in ra and dec coordinates
    
    q : list of paralactic angle for each observation
    
    Returns
    -------
    aaIxx, aaIyy, aaIxy : The moments in the new coordinate system [[I_alt_alt, I_alt_az],[I_alt_az, I_az_az]]
    """
    from lsst.afw.geom import Quadrupole
    from lsst.geom import LinearTransform
    
    cq, sq = np.cos(q), np.sin(q)
    R = [np.array([[cq[i], sq[i]], [-sq[i], cq[i]]]) for i in range(len(cq))]
    
    aaIxx = []
    aaIyy = []
    aaIxy = []
    
    for i, nwIxx_ in enumerate(nwIxx):
        shape = Quadrupole(nwIxx_, nwIyy[i], nwIxy[i])
        rotShape = shape.transform(LinearTransform(R[i]))    
        
        aaIxx.append(rotShape.getIxx()) 
        aaIyy.append(rotShape.getIyy()) 
        aaIxy.append(rotShape.getIxy())


    return np.array(aaIxx), np.array(aaIyy), np.array(aaIxy)