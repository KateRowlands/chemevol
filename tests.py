import pytest
from functions import remnant_mass, destruction_timescale, \
                    grow_timescale,metallicity, dust_masses, inflows, outflows
from lookups import lifetime, mass_yields, dust_mass_sn, find_nearest

class TestFunctions:

    def test_remnant_mass_mid(self):
        mass = remnant_mass(10.)
        assert mass.value == 1.5

    def test_remnant_mass_low(self):
        mass = remnant_mass(1.)
        assert 0.5515 < mass.value < 0.5525
        
    def test_remnant_mass_high(self):
        mass = remnant_mass(40.)
        assert 10.6 < mass.value <10.65

    def test_destruction(self):
        destroy = destruction_timescale(1000.,6e9,2.1e7)
        destroy = destroy/1e6 #in Myr
        assert  142.8 < destroy.value < 142.9
    
    def test_graingrowth(self):
        grow = grow_timescale(500.,3.35e9,1.169e9,6.64e-2,(0.671*6.64e-2))
        grow = grow/1e6 #in Myr
        assert 43.1 < grow.value < 43.2 
        
    def test_metallicity(self):
        metals = metallicity(8e8, 4e10)
        assert 0.019 < metals < 0.022
    
    def test_dustmass_low(self): 
        mass = dust_masses(0.45, 1.0, 1.0)
        assert mass.value == 0.45
    
    def test_dustmass_outofbounds_low(self):
        mass = dust_masses(0.45, 0.7, 1.0)  
        assert mass.value == 0.0  
    
    def test_dustmass_outofbounds_high(Self):
        mass = dust_masses(0.45, 60.0, 1.0) 
        assert mass.value == 0.0 
    
    def test_dustmass_mid(Self):
        mass = dust_masses(0.45, 25.0, 1.0)
        assert mass.value == 1.0  
    
    def test_outflow_func(Self):       
        gas_outflow = outflows(1.0,1.5)
        assert gas_outflow.value == 1.5
        
    def test_inflow_func(Self):       
       gas_inflow = inflows(1.0,1.5)
       assert gas_inflow.value == 1.5
       
class TestTables:
    
    def test_lifetimes(self):
        assert lifetime[1][1] == 9.5
        assert lifetime[15][2] == 0.0026        

    def test_yields_001(self):
        assert mass_yields[1][1] == 0.
        assert mass_yields[12][1] == 0.27   
        assert mass_yields[1][4] == 8.54e-4
        assert mass_yields[20][3] == 41.6 
        assert mass_yields[1][6] == 1.12e-4
        assert mass_yields[20][6] == 9.39 
        assert mass_yields[0][8] == 6.83e-3
        assert mass_yields[19][8] == 17.75 
        