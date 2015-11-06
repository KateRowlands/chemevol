'''
Chemevol - Python package to read in a star formation history file,
input galaxy parameters and run chemical evolution to determine the evolution
of gas, metals and dust in galaxies.

The code is based on Morgan & Edmunds 2003 (MNRAS, 343, 427)
and described in detail in Rowlands et al 2014 (MNRAS, 441, 1040).

If you use this code, please do cite the above papers.

Copyright (C) 2015 Haley Gomez and Edward Gomez, Cardiff University and LCOGT
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
'''

from astropy import units as u
import numpy as np
import logging

logger = logging.Logger('chem')

def ejected_gas_mass(m, sfr, choice):
    '''
    Calculate the ejected mass from stars by mass loss/stellar death
    at time t, needs to be integrated from mass corresponding to
    age of system (tau(m)) -- 120 Msolar
    
    de/dm = (m-m_R(m)) x SFR(t-tau(m)) x phi(m) 
    '''
    dej = (m - remnant_mass(m)) * sfr * initial_mass_function(m, choice)
    return dej

def metallicity(metalmass,gasmass):
    '''
    Calculates the metal mass fraction 
    
    Z = M_Z/M_g  
    '''
    metals = metalmass/gasmass
    return metals  

def remnant_mass(m):
    '''
    Calculates the remnant mass of a star of mass m.

    The formulism is based on Ferreras & Silk 2000 (ApJ 532 193)
    which in turn is based on Iben & Tsutukov 1984, Woosley & Weaver 1995.
    Stars with mass above m_bh don't eject material into the ISM
    '''
    if m <= 9.0:
        rem_mass = 0.106*m + 0.446
    elif (m > 9.0) & (m < 25.0):
        rem_mass = 1.5
    else:
        rem_mass = 0.61*m - 13.75
    rem_mass = rem_mass*u.solMass
    return rem_mass

def grow_timescale(e,G,SFR,Z,D):
    '''
    Calculates the grain growth timescale in years.

    - e: free parameter with MW value set to ~500.
         e = 2e4 would result in timescales of 1Myr and below.
    - G: gas mass in Msolar
    - D: dust mass in Msolar
    - Z: metallicity mass fraction
    - SFR: star formation rate in units of Msolar/Gyr

    Based on Mattsson & Andersen 2012 (MNRAS 423, 38)
    In dust evolution, dMd/dt is proportional to Md/t_grow
    '''

    SFR_in_years = SFR/1e9 # to convert from per Gyr to per yr
    t_grow = G/(e*Z*SFR_in_years)
    t_grow = t_grow/(1-((D/G)/Z)) #to account for metals already locked up in grains

    t_grow = t_grow*u.year
    return t_grow

def destruction_timescale(m,G,SN_rate):
    '''
    Calculates the dust destruction timescale in years.


    - SN_rate: supernova rate in N/Gyr
    - G: gas mass in Msolar
    - m: the amount of gas cleared by each supernova event.
    This is often 100 or 1000 Msolar, appropriate for SNe expanding
    into galactic densities of 1cm^-3 or 0.1cm^-3 respectively.

    Based on Dwek, Galliano & Jones 2004 (ApJ, 662, 927)
    In dust evolution, dMd/dt is proportional to Md/t_destroy
    '''

    SN_rate_in_years = SN_rate/1e9 # to convert from per Gyr to per yr
    t_destroy = G/(m*SN_rate_in_years)

    t_destroy = t_destroy*u.year
    return t_destroy

def initial_mass_function(m, choice='Chab'):
    '''
    Returns the IMF for a given choice of function and mass range.

    - "Chab", "chab" or "c" selects the Chabrier 2003 IMF (PASP 115 763)
    - "TopChab", "topchab", or "tc" selects a top heavy Chabrier IMF with high mass slope -0.8
    - "Kroup", "kroup" or "k" selects the Kroupa & Weidner 2003 IMF (ApJ 598 1076)
    - "Salp", "salp" or "s" selects the Salpeter 1955 IMF (ApJ 121 161)
    '''

    if (choice == "Chab" or choice == "chab" or choice == "c"):
        if m <= 0.5:
            imf = np.exp(-1.*(np.log10(m)-np.log10(0.079))**2.)
            imf = (0.85*imf)/((2.*0.69**2.))/m
        else:
            imf = 0.24*(m**-1.3)/m

    if (choice == "TopChab" or choice == 'topchab' or choice == "tc"):
        # If you want to do -0.5 slope, need to change norm factor by
        # 4.72424
        if m <= 1.0:
            imf = np.exp(-1.*(np.log10(m)-np.log10(0.079))**2.)
            imf = imf*(0.85/2.21896)/((2.*0.69**2.))/m
        else:
            imf = (0.24/2.21896)*(m**-0.8)/m

    if (choice == "Kroup" or choice == "kroup" or choice == "k"):
        if m <= 0.5:
            imf = 0.58*(m**-0.30)/m
        elif (m > 0.5) & (m <= 1.0):
            imf = 0.31*(m**-1.20)/m
        else:
            imf = 0.31*(m**-1.70)/m

    if (choice == "Salp" or choice == "salp" or choice == "s"):
        imf = (0.17/0.990465)*(m**-1.35)/m
    return imf

def validate_initial_dict(keysdict, data_dict):
    '''
    Validate the initial data

    example
    input = {
			'gasmass_init':4e10,
			'SFH':'MilkyWay.sfh',
			'gamma':0,
			'IMF_fn':'Chab',
			'dust_source':'ALL',
			'destroy':True,
			'inflows':0,
			'outflows':0
			}
    '''
    for run,keys in data_dict.items():
        for k in keysdict:
            try:
                dummy = data_dict[run][k]
            except KeyError:
                print("Oops key %r is missing in %r" % (k,run))
            else:

            # check gasmass, gamma, inflows and outflows are numbers:
                if (k == 'gasmass_init') or (k == 'inflows') \
                    or (k == 'outflows') or (k == 'gamma'):
                    try:
                        dummy = int(data_dict[run][k])
                    except ValueError:
                        print("Oops we were expecting a number in %r:%r" % (run,k))

            # check SFH is a string :
                if (k == 'SFH'):
                    if not isinstance(data_dict[run][k], basestring):
                        raise TypeError("Oops %r:%r should be a string" % (run,k))

            # check dust_source options correct
                if (k == 'dust_source'):
                    dummy = data_dict[run][k]
                    if not ((dummy == 'SN') or (dummy == 'LIMS') or \
                           (dummy == 'LIMS+SN') or (dummy == 'GG') or\
                           (dummy == 'ALL')):
                        raise ValueError("Oops double check %r:%r" % (run,k))
            
            # check IMF_fn source options correct
                if (k == 'IMF_fn'):
                    dummy = data_dict[run][k]    
                    if not ((dummy == 'Chab') or (dummy == 'chab') or (dummy == 'c') or  \
                            (dummy == 'TopChab') or (dummy == 'topchab') or (dummy == 'tc') or \
                            (dummy == 'Kroup') or (dummy == 'kroup') or (dummy == 'k') or \
                            (dummy == 'Salp') or (dummy == 'salp') or (dummy == 's')):
                        raise ValueError("Oops check %r in %r" % (k,run))
            