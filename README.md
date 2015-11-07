# chemevol
Chemical evolution python package

# Requirements

## Python packages
- numpy
- astropy
- logger

## input files needed
The code reads in a star formation history from a file called filename.sfh.  This needs to be in the form time (yr), SFR (Msolar/yr).    An example is provided `MilkyWay_yin.sfh` based on Yin et al 2009 (A & A, 505, 497).

## Running the code
```python
from evolve import ChemModel
inits = {	'gasmass_init':4.8e10,
							'SFH':'MilkyWay.sfh',
							'gamma':0,
							'IMF_fn':'Chab',
							'dust_source':'ALL',
							'destroy':True,
							'inflows':{'metals': 0., 'xSFR': 0},
							'outflows':{'metals': True, 'xSFR': 0},
							}
ch = ChemModel(**inits)
time, mgas = ch.gas_mass()
time, mstars = ch.stellar_mass()
```
