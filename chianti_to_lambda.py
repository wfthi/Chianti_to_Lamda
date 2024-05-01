"""
Convert the data in the Chianti database into the Leiden-Lamba format

The code requires that Chiantipy v >  (0.10.0) and the Chianti 
database (v 10.0) are installed. See
https://chiantipy.readthedocs.io/en/stable/getting_started.html

Please follow the instruction and ensure that chiantypy can access
the Chianti database.

https://www.chiantidatabase.org

Copyright (C) 2024  Wing-Fai Thi

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import ChiantiPy.core as ch
import ChiantiPy.tools.data as chdata
import numpy as np
import astropy.units as u
from astropy.constants import c as cl
from astropy.constants import h as hplanck
from astropy.constants import k_B as kbolt


chdata.Defaults.keys()
chdata.Defaults['wavelength']
path = chianti_database = os.environ['XUVTOP']
if path is not None:
    print('Path to the Chianti datbase found')
    pass
else:
    print('No path to the Chianti databse version > 10')


def get_levels(sp):
    """
    Extract the level information from Chianti sp object

    Parameter
    ---------
    sp : Chianti object

    Returns
    -------
    name_level: array of str
        level spectroscopic names

    weight: array of float
        statistical weights of the levels

    Elevel: array of float
        the energy levels in cm^-1

    jval; array of int
        the level quantum number j
    """
    # Energy levels in cm^-1
    Elevel = sp.Elvlc['ecm'] / u.cm
    weight = np.array(sp.Elvlc['mult'])
    name_level = np.array(sp.Elvlc['pretty'])
    jval = np.array(sp.Elvlc['j'])
    return name_level, weight, Elevel, jval


def get_spectroscopy(sp, Elevel):
    """
    Extract the spectropic data from Wgfa
    """
    ind_spectro_lower = np.array(sp.Wgfa['lvl1'])
    ind_spectro_upper = np.array(sp.Wgfa['lvl2'])
    name_spectro_lower = sp.Wgfa['pretty1']
    name_spectro_upper = sp.Wgfa['pretty2']
    spectro_wl = np.abs(sp.Wgfa['wvl']) * u.angstrom
    # Wavelengths calculated from the theoretical energies are of an
    # indeterminate accuracy and their values are presented as negative
    # values of the calculated wavelength.
    # wtheo = spectro_wl < 0. * u.angstrom
    EinsA = sp.Wgfa['avalue'] / u.s
    spectro_freq = (cl / spectro_wl).to(u.Hz)
    Eupper = ((Elevel[ind_spectro_lower - 1] * cl + spectro_freq) *
              hplanck / kbolt).to(u.K)
    return ind_spectro_lower, ind_spectro_upper, name_spectro_lower, \
        name_spectro_upper, spectro_freq, EinsA, Eupper


def output_levels_transitions(lambda_name, sp, mass, Elimit, Elevel,
                              Eupper, weight, name_level, ind_spectro_upper,
                              ind_spectro_lower, EinsA, spectro_freq, jval):
    """
    Write the levels and transitions in the Leiden-Lambda format
    """
    print(lambda_name)
    print('Level energy limit ', Elimit)
    # kT/hc -> cm^-1
    Elimit_wavenumber = (Elimit * kbolt / hplanck / cl).to(u.cm**(-1))
    # print('Level energy limit in wavenumber', Elimit_wavenumber)
    if sp.Ion == 1:
        the_file.write('!ATOM\n')
        the_file.write(lambda_name + '\n')
        the_file.write('ATOM WEIGHT\n')
    else:
        the_file.write('!ION\n')
        the_file.write(lambda_name + '\n')
        the_file.write('ION WEIGHT\n')
    the_file.write(f'{mass}\n')
    # print('Maximum level energy ', Elevel.max())
    wlevel = Elevel <= Elimit_wavenumber
    nb_level = len(Elevel[wlevel])
    the_file.write('!NUMBER OF ENERGY LEVELS\n')
    the_file.write(f'{nb_level}\n')
    the_file.write('!LEVEL + ENERGIES(cm^-1) + WEIGHT + J\n')
    for level, (energ, glevel, name, jv) in enumerate(zip(Elevel,
                                                          weight,
                                                          name_level, jval)):
        if energ <= Elimit_wavenumber:
            the_file.write(f'{level+1:4} {energ.value:.3f}\
                           {glevel:.2f} {jv:.1f} ! {name}\n')
    the_file.write('!NUMBER OF RADIATIVE TRANSITIONS\n')
    wlevel = Eupper <= Elimit  # Kelvin
    nb_level = len(Eupper[wlevel])
    the_file.write(f'{nb_level}\n')
    header = '!TRANS + UP + LOW + EINSTEINA(s^-1) + FREQ(GHz) + E_u(K)\n'
    the_file.write(header)
    lv = 0
    for upper, lower, A, f, Eu in zip(ind_spectro_upper, ind_spectro_lower,
                                      EinsA, spectro_freq, Eupper):
        if Eu <= Elimit:
            lv += 1
            the_file.write(f'{lv:4d} {upper:4d} {lower:4d}  {A.value:12.4E} {1e-9*f.value:.3f}  {Eu.value:.4f}\n')


def get_rate_coeff_electron(sp):
    """
    Get the rate_coefficients in collisions with electrons from sp
    """
    ind_coll_lower = sp.Scups['lvl1']
    ind_coll_upper = sp.Scups['lvl2']
    sp.upsilonDescale(prot=0)  # electron as collision partner
    rate_coeff_electron = sp.Upsilon['dexRate']
    return ind_coll_lower, ind_coll_upper, rate_coeff_electron


def get_rate_coeff_proton(sp):
    """
    Get the rate_coefficients in collisions with protons from sp
    """
    if sp.Npsplups > 0:
        print('Collision rate coefficients with proton for', sp.Spectroscopic)
        sp.upsilonDescale(prot=1)
        rate_coeff_proton = sp.Upsilon['dexRate']
        return rate_coeff_proton
    else:
        print('No collision rate coefficients with proton for',
              sp.Spectroscopic)
        return None


def output_collision_rates(name, ind_limit, ind_coll_lower,
                           ind_coll_upper, rate_coeff, temp):
    """
    Output the rates in the leiden-Lamba format
    """
    the_file.write('!COLLISIONS BETWEEN\n')
    the_file.write(name + ' ! CHIANTI v10\n')
    the_file.write('!NUMBER OF COLL TRANS\n')
    wlimit = ind_coll_upper <= ind_limit
    nb_coll_trans = len(rate_coeff[wlimit])
    the_file.write(f'{nb_coll_trans}\n')
    the_file.write('!NUMBER OF COLL TEMPS\n')
    nb_temp = len(temp)
    the_file.write(f'{nb_temp}\n')
    the_file.write('!COLL TEMPS\n')
    str_temp = str(temp[0])
    for i in range(1, len(temp)):
        str_temp += ' ' + str(temp[i])
    the_file.write(str_temp + '\n')
    the_file.write('!TRANS + UP + LOW + COLLRATES(cm^3 s^-1)\n')
    #
    trans = 0
    for lo, up, rate in zip(ind_coll_lower, ind_coll_upper, rate_coeff):
        if up <= ind_limit:
            trans += 1
            str_rate = str(f'{rate[0]:12.4E}')
            for i in range(1, len(rate)):
                str_rate += ' ' + str(f'{rate[i]:12.4E}')
            the_file.write(f'{trans:4d} {up:4d} {lo:4d} {str_rate}\n')


def write_lambda_file(species_name, lambda_name, mass, mssg1, mssg2,
                      Elimit, temp, filename):
    """
    General routine to read the Chianti databse using chiantipy
    and output a ascii file in the Leiden Lamba format

    Parameters
    ----------
    species_name:
        the name of the species within Chianti

    lambda_name: str
        the name of the species for Lamba

    mass: float
        the mass of the species in amu

    mssg1, mssg2: str
        the header information for each collision
        one for the collisions with electron
        one for the collisions with protons
        The header follows the Lamba format, for example
            mssg1 = '1 Na++ + e'
            mssg2 = '2 Na++ + H+'

    Elimit: astropy.Quantity unist cm^-1
        the upper energy format in Kelvin

    temp: array of floats
        the temperatures at which the collisional data are provided

    filename: str
        full path of the output file in the lamba format

    Returns
    -------
    : None
        the code write a file on the disk

    Examples
    --------
    >>> species_name = 'fe_3'
    >>> lambda_name = 'Fe++'
    >>> mssg1 = '1 Fe++ + e'
    >>> mssg2 = '2 Fe++ + H+'
    >>> mass = '55.845'
    >>> filename = 'FeIII_lambda_chianti.dat'  # Lamba file output name
    >>> Elimit = 1e5 * u.K
    >>> # Call the main routine
    >>> write_lambda_file(species_name, lambda_name, mass, mssg1, mssg2,
    ...                   Elimit, temp, filename)
    """
    sp = ch.ion(species_name, temperature=temp)
    sp.Spectroscopic
    name_level, weight, Elevel, jval = get_levels(sp)

    ind_spectro_lower, ind_spectro_upper, _,\
        _, spectro_freq, EinsA, Eupper = get_spectroscopy(sp, Elevel)

    ind_coll_lower, ind_coll_upper, rate_coeff_electron =\
        get_rate_coeff_electron(sp)
    rate_coeff_proton = rate_coeff_proton = get_rate_coeff_proton(sp)

    global the_file
    with open(filename, 'w') as the_file:
        output_levels_transitions(lambda_name, sp, mass, Elimit, Elevel,
                                  Eupper, weight, name_level,
                                  ind_spectro_upper,
                                  ind_spectro_lower, EinsA, spectro_freq, jval)
        the_file.write('!NUMBER OF COLL PARTNERS\n')
        w = Eupper < Elimit
        if np.count_nonzero(w) > 0:
            ind_limit = ind_spectro_upper[Eupper < Elimit].max()
            if rate_coeff_proton is not None:
                the_file.write('2\n')  # 2 collision partners
            else:
                the_file.write('1\n')  # 2 collision partners
            output_collision_rates(mssg1, ind_limit, ind_coll_lower,
                                   ind_coll_upper, rate_coeff_electron, temp)
            if rate_coeff_proton is not None:
                output_collision_rates(mssg2, ind_limit, ind_coll_lower,
                                       ind_coll_upper, rate_coeff_proton, temp)
        else:
            print("No level found")
            print("Increase the value of Elimit")
    the_file.close()
    print("Done writting the file ", filename)


if __name__ == "__main__":
    # First example Na++
    # the temperature output
    temp = np.array([10.0, 20.0, 50.0, 100.0, 200, 500, 1000.0, 2000.0,
                     10000., 20000.])
    # Chianti name
    species_name = 'na_3'
    # Lamba name
    lambda_name = 'Na++'
    mssg1 = '1 Na++ + e'  # 1 for first collision partner (lamba format)
    mssg2 = '2 Na++ + H+'
    # mass in amu
    mass = '22.989769'
    # lamba output file name
    filename = 'NaIII_lambda_chianti.dat'
    # Upper limit for the level energy in Kelvin (astropy Quantity)
    Elimit = 1e5 * u.K
    # Call the main routine
    write_lambda_file(species_name, lambda_name, mass, mssg1, mssg2,
                      Elimit, temp, filename)
    # ----------------------------------------------------------------
    # Second example Fe++
    species_name = 'fe_3'
    lambda_name = 'Fe++'
    mssg1 = '1 Fe++ + e'
    mssg2 = '2 Fe++ + H+'
    mass = '55.845'
    filename = 'FeIII_lambda_chianti.dat'  # Lamba file output name
    Elimit = 1e5 * u.K
    # Call the main routine
    write_lambda_file(species_name, lambda_name, mass, mssg1, mssg2,
                      Elimit, temp, filename)
    # ----------------------------------------------------------------
    # Example 3 Ca+
    species_name = 'ca_2'  # Chianti uses lower case
    lambda_name = 'Ca+'
    mssg1 = '1 Ca+ + e'
    mssg2 = '2 Ca+ + H+'
    mass = '40.078'  # please use quotes to make it a string
    filename = 'CaII_lambda_chianti.dat'  # Lamba file output name
    Elimit = 1e5 * u.K
    # Call the main routine
    write_lambda_file(species_name, lambda_name, mass, mssg1, mssg2,
                      Elimit, temp, filename)
    # ----------------------------------------------------------------
    # Example 4 Ne+
    species_name = 'ne_2'  # Chianti uses lower case
    lambda_name = 'Ne+'
    mssg1 = '1 Ne+ + e'
    mssg2 = '2 Ne+ + H+'
    mass = '20.1797'  # please use quotes to make it a string
    filename = 'NeII_lambda_chianti.dat'  # Lamba file output name
    Elimit = 1e5 * u.K
    # Call the main routine
    write_lambda_file(species_name, lambda_name, mass, mssg1, mssg2,
                      Elimit, temp, filename)
