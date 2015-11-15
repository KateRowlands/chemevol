#cython: boundscheck=False, wraparound=False, nonecheck=False
cimport numpy as np
import numpy as np
import functions as f
from lookups import lookup_taum, lookup_fn, t_lifetime

def dust_integral(chem, snrate_l, gasmass_l, metallicity_l, time_l):
    # initialize
    cdef double md = 0.
    cdef double md_all = 0.
    cdef double md_stars = 0.
    cdef double md_gg = 0.
    cdef double prev_t = 1e-3
    cdef double t, mg, z, r_sn, mdust_stars, mdust_inf, mdust_out
    cdef double mdust_gg, mdust_des
    cdef double t_gg, t_des, dt
    cdef double ddust1, ddust2, dust_source_all
    cdef double[:] time = time_l
    cdef int length = len(time)
    cdef double[:] snrate = snrate_l
    cdef double[:] metallicity = metallicity_l
    cdef double[:] gasmass = gasmass_l
    cdef double[:] dust_list = np.zeros(length)
    cdef double[:] dz_ratio_list = np.zeros(length)
    timescales = []
    dust_list_sources = []
    for item in range(length):
        t = time[item]
        mg = gasmass[item]
        z = metallicity[item]
        r_sn = snrate[item]

    #set up dust mass from stars (recycled(LIMS) + new (SN+LIMS))
        mdust_stars = ejected_d_mass(chem,t, z)

    # set up inflow contribution to dust mass (read from dictionary)
        mdust_inf = chem.inflows['dust']*f.inflows(chem.sfr(t), chem.inflows['xSFR']).value

        if chem.outflows['dust']:
            mdust_out = (1./mg)*f.outflows(chem.sfr(t), chem.outflows['xSFR']).value
        else:
            mdust_out = 0.

        # destruction timescales + dust mass from grain growth and destruction
    #    t_des = 1e-6*f.destruction_timescale(chem.destroy_ism,mg,r_sn).value
        mdust_gg, t_gg = f.graingrowth(chem.epsilon,mg,chem.sfr(t),z,md,chem.coldfraction)
        mdust_des, t_des = f.destroy_dust(chem.destroy_ism,mg,r_sn,md,chem.coldfraction)
    # Integrate dust mass equation with time
        ddust1 = - md*f.astration(mg,chem.sfr(t))
        ddust2 =    + mdust_stars \
                    + mdust_inf \
                    - md*mdust_out \
                    + mdust_gg \
                    - mdust_des
        # to plot dust sources (these are dust in rather than dust mass at any time)
        dust_source_all = mdust_stars + mdust_gg
        dt = t - prev_t
        prev_t = t
        md += (ddust1+ddust2)*dt
        md_all += dust_source_all*dt
        md_gg += mdust_gg*dt
        md_stars += mdust_stars*dt
        dust_list[item] = md
        dust_list_sources.append((md_all, md_stars, md_gg))
        # save timescales for grain growth and destruction in Gyr
        timescales.append((t_des,t_gg))
        if z <= 0.:
            dust_to_metals = 0.
        else:
            dust_to_metals = (md/mg)/z
    #    print t, mg/4.8e10, z, mdust_des, t_des, t_gg #mdust_ast, mdust_stars, des
        dz_ratio_list[item] = dust_to_metals
    return dust_list, dust_list_sources, dz_ratio_list, timescales

def ejected_d_mass(chem, double t, double metallicity):
    '''
    Calculates the ejected dust mass from stars edm (t)
    for dust mass integral where:
    dmd/dt = - md/mg * SFR(t) + int edm*dm + md/mg,inflows - md/mg,outflows
             + md_graingrowth - md_destroy
    '''
    # initialize
    cdef double dm = 0.01
    cdef double edm = 0.
    cdef double mu = t_lifetime[-1][0]
    # we pull out mass corresponding to age of system
    # to get lower limit of integral
    cdef double m = lookup_fn(t_lifetime,'lifetime_low_metals',t)[0]
    cdef double tdiff, zdiff, taum,sfr_diff
    #to make taum lookup faster
    lifetime_cols = {'low_metals':1, 'high_metals':2}
    if metallicity < 0.019:
        col_choice = lifetime_cols['low_metals']
    else:
        col_choice = lifetime_cols['high_metals']
    while m <= mu:
        if m > 10.:
            dm = 0.5
        # pull out lifetime of star of mass m so we can
        # calculate SFR when star was born which is t-lifetime
        taum =  lookup_taum(m,col_choice)
        tdiff = t - taum
        if tdiff <= 0.:
            zdiff = 0.
            sfr_diff = 0.
        else:
            zdiff = metallicity #needs to be z(t-taum)
            sfr_diff = chem.sfr(tdiff)
        edm += f.ejected_dust_mass(m, sfr_diff, zdiff, metallicity, chem.imf_type) * dm
        m += dm
#            print m, t, tdiff, em
    return edm
