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
import functions as f
import plots as p
import matplotlib.pyplot as plt

'''------------------------------------------------------------------------
First set up initial parameters for galaxy model by editing the dictionary
initial_galaxy_params

- gasmass_init: 		initial gas mass in solar masses
- SFH: 					filename for star formation history file (filename.sfh)
						if you don't specify a file, it will default to MW like SFH
- gamma: 				power law if using SFH generated by MAGPHYS code
- IMF_fn:          		choice of IMF function: Chab/chab/c, TopChab/topchab/tc,
		  				Kroup/kroup/k or Salp/salp/s
- dust_source: 			choice of dust sources to be included:
						SN: supernova dust onlt
						LIMS: low intermediate mass stars dust only
						LIMS+SN: both SN and LIMS included
						GG: interstellar grain growth only
						ALL: grain growth
- destroy: 				add dust destruction from SN shocks True or False
- inflows: 				there are two parameters
 						metals = metallicity of inflow: input a number
						xSFR = inflow rate is X * SFR: input a number
						dust = amount of dust inflow: input a number
- outflows: 			there are two parameters
 						metals = metallicity of inflow: input True or False
								 True = metallicity of system, False = 0
						xSFR = inflow rate is X * SFR: input a number
						dust = amount of dust in outflow: input True of False
							  	 True = dust/gas of system, False = 0
- cold_gas_fraction = 	fraction of gas in cold dense state for grain growth
					  	typically 0.5-0.9 for high z systems, default is 0.5
- epsilon_grain = 		grain growth parameter from Mattsson & Andersen 2012
						default is 500 for t_grow ~ 10Myr.
- destruct = 			amount of material destroyed by each SN (typically 1000 or 100Msun)


Each run will be used to generate the evolution of dust, gas,
SFR, metals and stars over time
---------------------------------------------------------------------------
'''

init_keys = (['gasmass_init','SFH','inflows','outflows','dust_source','destroy',\
			'IMF_fn','gamma','epsilon_grain','destruct'])

initial_galaxy_params = {'run1': {
							'gasmass_init': 4.8e10,
							'SFH': 'MilkyWay.sfh',
							'gamma': 0,
							'IMF_fn': 'Chab',
							'dust_source': 'ALL',
							'destroy':True,
							'inflows':{'metals': 0., 'xSFR': 0, 'dust': True},
							'outflows':{'metals': True, 'xSFR': 0, 'dust': False},
							'cold_gas_fraction': 0.5,
							'epsilon_grain': 500.,
							'destruct': 1000.
							},
						'run2': {
							'gasmass_init':4e10,
							'SFH':'MilkyWay.sfh',
							'gamma':0,
							'IMF_fn':'c',
							'dust_source':'GG',
							'destroy':True,
							'inflows':{'metals': 1e-4, 'xSFR': 0, 'dust': 0},
							'outflows':{'metals': False, 'xSFR': 0, 'dust': False},
							'cold_gas_fraction':0.,
							'epsilon_grain': 500.,
							'destruct': 100.
							}
							}

#Now we will test that the input parameters are A-OK:
#f.validate_initial_dict(init_keys, initial_galaxy_params)

from evolve import ChemModel
ch = ChemModel(**inits)

time, mgas = ch.gas_mass()
time, mstars = ch.stellar_mass()
time, metalmass, metallicity = ch.metal_mass(mgas)
snrate = ch.supernova_rate()
time, mdust, dust_to_metals = ch.dust_mass(mgas,metallicity,snrate)

gasfraction = mgas/(mgas+mstars)

# Quick look plots:
p.figure(time,mgas,mstars,metalmass,metallicity,mdust,dust_metals)
