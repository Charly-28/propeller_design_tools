import os
import subprocess
import shutil
import matplotlib.pyplot as plt
import numpy as np
from propeller_design_tools.airfoil import Airfoil
from propeller_design_tools.radialstation import RadialStation
from propeller_design_tools.propeller import Propeller
from propeller_design_tools.user_io import Info, Error
from propeller_design_tools.user_settings import _get_user_settings, get_prop_db


# =============== CONVENIENCE / UTILITY FUNCTIONS ===============
def delete_propeller(prop, verbose: bool = True):
    fpaths = [prop.xrr_file, prop.xrop_file, prop.meta_file]
    rmvd = []
    for path in fpaths:
        if os.path.isfile(path):
            os.remove(path)
            rmvd.append(path)
    if verbose:
        Info('Removed paths: {}'.format(rmvd))


def clear_foil_database(inside_root_db: bool = False, inside_polar_db: bool = True, inside_for_xfoil: bool = True):
    """
    Helper function to clear out files from foil database.

    :param inside_root_db:
    :param inside_polar_db:
    :param inside_for_xfoil:
    :return:
    """
    if inside_root_db:
        delete_files_from_folder(_get_user_settings()['airfoil_database'])
    if inside_polar_db:
        delete_files_from_folder(os.path.join(_get_user_settings()['airfoil_database'], 'polar_database'))
    if inside_for_xfoil:
        delete_files_from_folder(os.path.join(_get_user_settings()['airfoil_database'], 'for_xfoil'))


def clear_prop_database(inside_root_db: bool = True, inside_xrotor_files: bool = True, inside_op_files: bool = True):
    """
    Helper function to clear out files from prop database.

    :param inside_op_files:
    :param inside_root_db:
    :param inside_xrotor_files:
    :return:
    """
    if inside_root_db:
        delete_files_from_folder(_get_user_settings()['propeller_database'])
    if inside_xrotor_files:
        delete_files_from_folder(os.path.join(_get_user_settings()['propeller_database'], 'xrotor_geometry_files'))
    if inside_op_files:
        delete_files_from_folder(os.path.join(_get_user_settings()['propeller_database'], 'xrotor_op_files'))


def delete_files_from_folder(folder: str):
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            fpath = os.path.join(folder, filename)
            if os.path.isfile(fpath):
                os.remove(fpath)
    else:
        raise FileExistsError('Cannot find folder: {}'.format(folder))


def search_files(folder: str, search_strs: list = None, contains_any: bool = False, include_dirs: bool = False):
    """
    Utility function to help users interface with getting files from folders

    :param folder: the full path to the folder to look in
    :param search_strs: a list of strings to search for in each filename
    :param contains_any: If True will return files that contain ANY of the search strings, defaults to False (returns only
        filenames that contain ALL of the search strings)
    :return: a list of filenames
    """

    if not os.path.exists(folder):
        raise FileExistsError('PDT ERROR: No folder named "{}" found!'.format(folder))

    if include_dirs:
        files = os.listdir(folder)
    else:
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    if search_strs == []:
        search_strs = None

    if search_strs is not None:
        # for s in search_strs:
        #     files = []
        if contains_any:  # ANY
            files = [f for f in files if any([s in f for s in search_strs])]
        else:  # ALL
            files = [f for f in files if all([s in f for s in search_strs])]

    return files


def get_airfoil_file_from_db(foil_name: str, exact_namematch: bool = False):
    db_dir = _get_user_settings()['airfoil_database']
    possible_files = search_files(folder=db_dir, search_strs=[foil_name])
    exact_fname = '{}.dat'.format(foil_name)

    if len(possible_files) == 1:
        if not exact_namematch:
            return possible_files[0]
        else:   # must match exactly
            if possible_files[0] == exact_fname:
                return possible_files[0]
            else:
                raise Error('Did not find an exact match for "{}"'.format(exact_fname))

    elif len(possible_files) == 0:
        raise Error('Did not find any coordinate files containing name "{}" when looking in user-set airfoil '
                    'database: {}'.format(foil_name, db_dir))

    else:   # found multiple files
        if not exact_namematch:
            raise Error('Found multiple coordinate files looking for airfoil "{}"!\n{}\n'
                        'Consider using kwarg exact_namematch=True'.format(foil_name, possible_files))
        else:
            possible_files = [f for f in possible_files if f == exact_fname]
            if len(possible_files) == 0:
                raise Error('Did not find any coordinate files exactly matching "{}"'.format(exact_fname))
            else:
                return possible_files[0]


def merge_polar_data_dicts(new: dict, old: dict):
    new_pol_keys = new[list(new.keys())[0]].keys()
    merged = new.copy()  # merged starts out as copy of the new dict
    for old_pol_key, old_pol in old.items():  # cylce thru old dict
        if old_pol_key not in new.keys():  # if saved (re, mach, ncrit) data not in new data
            if old_pol.keys() == new_pol_keys:  # if keys are the same (alpha, CL, CD, etc)
                merged[old_pol_key] = old_pol  # append and move on
            else:  # if keys aren't same, warn that data isn't being retained
                print('Warning: found saved polar data of older/incorrect format -> deleting {}'.format(old_pol_key))
        else:  # if saved (re, mach, ncrit) data in new data
            if old_pol.keys() == new_pol_keys:  # if same keys (alpha, CL, CD, etc) inform of overwrite
                pass
                # print('Warning: overwriting saved polar data for {}'.format(old_pol_key))
            else:  # if not same keys
                print('Warning: overwriting saved polar data for {} (was older/incorrect format)'.format(old_pol_key))
    return merged


def scrub_nans(d: dict):
    # get all the indices where there's nans
    nan_idxs = []
    for key, val in d.items():
        if hasattr(val, '__len__'):     # only consider arrays / lists
            idxs = np.where(np.isnan(val))[0]
            nan_idxs.extend(list(idxs))
    nan_idxs = list(set(nan_idxs))

    # now cycle through and create a new dictionary without nans
    new_d = {}
    for key, val in d.items():
        if isinstance(val, list):   # if it's a list, return a list
            new_d[key] = [v for i, v in enumerate(val) if i not in nan_idxs]
        elif isinstance(val, np.ndarray):   # if it's a np array, return an np array
            new_d[key] = np.array([v for i, v in enumerate(val) if i not in nan_idxs])
        else:   # otherwise, just stuff it back in to the new dictionary un-altered
            new_d[key] = val

    return new_d


# =============== FILE READING / WRITING FUNCTIONS ===============
def read_airfoil_coordinate_file(fpath: str, verbose: bool = True):
    """
    Function to read an airfoil coordinate listing file in .dat or .txt format, assumes foil name is on
    1st line, then reads in the list of coordinates following (ignoring any lines with numbers > 1.0, and
    assuming the coordinates are single-space delimited) -> this is the file format as downloaded from

    https://m-selig.ae.illinois.edu/ads/coord_database.html

    It then splits the coordinates into upper and lower, and determines if either needs to be reversed
    (in order to read into xfoil right and also plot correctly), then reverses them if needed and returns
    arrays of x, y coords.

    :param
        fpath: the full filepath to the coordinate file
    :return
        name: the name of the airfoil from line1
        x_coords: np.array of x-coordinates
        y_coords: np.array of y-coordinates
    """

    # open the file and read contents
    with open(fpath, 'r') as f:
        txt = f.read()

    # prep arrays
    x_coords = np.array([])
    y_coords = np.array([])

    # split the text on returns and iterate over the lines
    lines = txt.strip(' ').split('\n')
    for i, line in enumerate(lines):
        # assumes name of airfoil is on line 0
        if i == 0:
            name = line.strip(' ')
        else:
            if not line.strip(' ') == '':  # skips blank lines
                vals = line.strip(' ').split(' ')  # split text on blank spaces
                x = float(vals[0])  # x coord is the 0th element
                y = float(vals[-1])  # y coord is the last element
                if x <= 1:  # only append if x in the range [0,1]
                    x_coords = np.append(x_coords, x)
                if y <= 1:  # only append if y in the range [0,1]
                    y_coords = np.append(y_coords, y)

    # make lists of coordinates, upper, and lower
    coords = np.array(list(zip(x_coords, y_coords)))
    x_directions = np.sign(np.diff(x_coords))
    x_directions = np.append(x_directions[0], x_directions)
    upper_end_idx = np.where(np.abs(np.diff(x_directions)) == 2)[0][0]
    upper_coords = coords[0:upper_end_idx + 1]
    lower_coords = coords[upper_end_idx + 1:]

    # flip upper if needed such that it is always decreasing in x
    if not x_directions[0] == -1:
        upper_coords = np.array(list(reversed(upper_coords)))

    # flip lower if needed such that it is always increasing in x
    if not x_directions[-1] == 1:
        lower_coords = np.array(list(reversed(lower_coords)))

    # check TE gap
    te_gap = upper_coords[0][1] - lower_coords[-1][1]
    te_gap_lim = 0.004
    blend_start = 0.7
    if te_gap < te_gap_lim:
        total_dy = (te_gap_lim - te_gap) / 2

        upper_idxs = np.where(upper_coords[:, 0] > blend_start)[0]
        for i in upper_idxs:
            xc, yc = upper_coords[i]
            dy = total_dy * (xc - blend_start) / (1 - blend_start)
            yc += dy
            upper_coords[i] = [xc, yc]

        lower_idxs = np.where(lower_coords[:, 0] > 0.8)[0]
        for i in lower_idxs:
            xc, yc = lower_coords[i]
            dy = total_dy * (xc - blend_start) / (1 - blend_start)
            yc -= dy
            lower_coords[i] = [xc, yc]
        if verbose:
            Info('Detected airfoil ({}) with TE thinner than hardcoded limit\n-> artificially adjusted TE gap to {} '
                     '(normalized)'.format(name, te_gap_lim))

    # re-combine upper and lower coords and split back into x, y arrays
    if upper_coords[-1][0] == lower_coords[0][0] and upper_coords[-1][1] == lower_coords[0][1]:  # check for double 0, 0
        coords = np.append(upper_coords, lower_coords[1:], axis=0)
    else:
        coords = np.append(upper_coords, lower_coords, axis=0)

    x_coords, y_coords = zip(*coords)
    return name, np.array(x_coords), np.array(y_coords)


def run_xfoil(foil_relpath: str, re: float, alpha: list = None, cl: list = None, iter_limit: int = 40, ncrit: int = 9,
              mach: float = 0.0, output_fpath: str = None, keypress_iternum: int = 1, tmout: int = 12,
              hide_windows: bool = True, verbose: bool = False):
    if not output_fpath:
        output_fpath = 'polar_output.txt'

    # swept pacc param
    if alpha is not None and cl is not None:
        print('Error: cannot give "run_xfoil" both "alpha" and "cl"')
        return
    elif alpha is not None:
        swept_param = {'alpha': alpha}
    elif cl is not None:
        swept_param = {'cl': cl}
    else:
        print('Error: must give "run_xfoil" either a list of "alpha" to sweep or a list of "cl" to sweep')
        return

    # sort the swept values
    vals = list(sorted(list(swept_param.values())[0]))

    # directory and temp command file, also xfoil path
    xfoil_dir = _get_user_settings()['airfoil_database']
    xfoil_cmnd_file = os.path.join(xfoil_dir, 'xfoil_inputs_temp.txt')
    xfoil_fpath = os.path.join(xfoil_dir, 'xfoil.exe')

    # write the command file
    with open(xfoil_cmnd_file, 'w') as f:
        f.write('load {}\n'.format(foil_relpath))
        f.write('ppar\nN\n200\n\n\n')
        f.write('oper\n')
        f.write('visc\n')
        f.write('{0:.0f}\n'.format(re))
        f.write('m {}\n'.format(mach))
        f.write('vpar\n')
        f.write('n {}\n\n'.format(ncrit))
        f.write('iter\n')
        f.write('{0:.0f}\n'.format(iter_limit))
        f.write('pacc\n\n\n')
        if 'alpha' in swept_param:
            for a in vals:
                f.write('a{}\n'.format(a))
        elif 'cl' in swept_param:
            for cl in vals:
                f.write('cl{}\n'.format(cl))
        f.write('!\n{}'.format(' ' * 100) * keypress_iternum)
        f.write('pwrt\n')
        f.write('{}\n\n\n'.format(output_fpath))
        f.write('quit\n')

    # now open it again and send it as the xfoil commands
    with open(xfoil_cmnd_file, 'r') as f:
        sui = subprocess.STARTUPINFO()
        if hide_windows:
            sui.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        if verbose:
            out, err = None, None
        else:
            out, err = subprocess.DEVNULL, subprocess.DEVNULL

        subprocess.run([xfoil_fpath], startupinfo=sui, stdin=f, stdout=out, stderr=err,
                       timeout=tmout, cwd=xfoil_dir)

    # delete the temp command file
    os.remove(xfoil_cmnd_file)


def read_xfoil_pacc_file(fpath: str = None, delete_after: bool = False):
    # if not given, default fpath is this
    if not fpath:
        fpath = os.path.join(_get_user_settings()['airfoil_database'], 'polar_output.txt')

    # open fpath and read in all text
    with open(fpath, 'r') as f:
        txt = f.read()

    # split into lines, create a dictionary, and iterate over the lines
    lines = [line for line in txt.split('\n') if line.strip() != '']
    d = {}
    for i, line in enumerate(lines):
        if i == 1:  # foil name on line 1
            d['name'] = line.split(':')[1].strip()
        elif i == 4:  # mach, re, and ncrit on line 4
            mach, rest = line.split('Mach')[1].split('Re')
            d['mach'] = float(mach.replace('=', '').strip())
            re, ncrit = rest.split('Ncrit')
            d['re'] = int(float(re.replace('=', '').replace(' ', '')))
            d['ncrit'] = int(float(ncrit.replace('=', '').strip()))
        elif i == 5:  # variable names on line 5, create an empty list for each one
            var_names = [name for name in line.strip().split(' ') if name != '']
            for key in range(len(var_names)):
                d[key] = []
        elif i >= 7:  # rest of lines in file contain variable entries for each variable
            vals = [val for val in line.strip().split(' ') if val != '']
            for col, val in enumerate(vals):
                if vals[0] not in d[0]:  # if the alpha value of this line is not already in the list of alphas
                    d[col].append(float(val))  # store with key as value index <- why am i doing it like this tho?
                else:
                    print('did not append over duplicate in col {} on line {}'.format(col, i - 7))

    # re-assign dictionary keys to be variable names instead of indexes
    for i, key in enumerate(var_names):
        d[key] = d[i]
        del d[i]

    if [len(v) for v in d.values() if isinstance(v, list)][0] == 0:
        # raise ValueError('XFOIL did not save any pacc data')
        return None

    # sort each variable based on alpha and re-assign to each dictionary key
    sort_list = d['alpha'].copy()
    for key in var_names:
        a, sorted_vals = list(zip(*sorted(zip(sort_list, d[key]))))
        d[key] = np.array(sorted_vals)

    # calculate L/D
    d['CL/CD'] = d['CL'] / d['CD']

    # if desired, delete the pacc file
    if delete_after:
        os.remove(fpath)

    return d


def read_polar_data_file(fpath: str):
    with open(fpath, 'r') as f:
        txt = f.read()

    # create a dictionary and go thru all the lines and populate dictionary of strings
    polar_data = {}
    lines = txt.split('\n')
    for i, line in enumerate(lines):
        if i == 0:
            name = line
        elif not i == 1:
            if '*****' in line:
                d = {'name': name}
                pairs = line.strip('*').strip().split(', ')
                for pair in pairs:
                    key, val = pair.split(' = ')
                    d[key.lower()] = val
            elif not line.strip() == '':
                key, val = line.split(': ')
                d[key] = val
            else:
                polar_data[(d['re'], d['mach'], d['ncrit'])] = d

    # convert all the strings into what they're supposed to be
    for str_key, polar in list(polar_data.items()):
        re_key, mach_key, ncrit_key = str_key
        for key, val in polar.items():
            if '(' in val:
                vals = [float(i.strip()) for i in val.replace('(', '').replace(')', '').split(',')]
                polar[key] = np.array(vals)
            elif 're' in key or 'ncrit' in key:
                polar[key] = int(val)
            elif 'mach' in key:
                polar[key] = float(val)
        new_key = (int(float(re_key)), float(mach_key), int(float(ncrit_key)))
        polar_data[new_key] = polar.copy()
        del polar_data[str_key]

    return polar_data


def save_polar_data_file(polar_data: dict, savepath: str, name: str = None):
    if not name:
        name = 'unnamed airfoil'
    with open(savepath, 'w') as f:
        f.write('{}\n'.format(name))
        for polar_key in polar_data:
            d = polar_data[polar_key]
            re, mach, ncrit = polar_key
            f.write('\n***** Re = {}, Mach = {}, nCrit = {} *****\n'.format(re, mach, ncrit))
            keys_2_write = [key for key, val in d.items() if isinstance(val, np.ndarray)]
            for key in keys_2_write:
                f.write('{}: {}\n'.format(key, tuple(d[key])))
    return


def read_xrotor_op_file(fpath):
    with open(fpath, 'r') as f:
        txt = f.read()
    if '********** NOT CONVERGED **********' in txt:
        Error('XROTOR did not converge')
    lines = txt.split('\n')
    d = {}
    headers = None

    for line in lines:
        if line.strip('-').strip('=').strip() == '':
            pass  # ignore blank lines
        elif ':' in line:
            spl = [l.strip() for l in line.split(':')]
            for i in range(len(spl) - 1):
                key = spl[i].split('  ')[-1].strip()
                val = spl[i + 1].split('  ')[0].strip()
                if val[0].isnumeric():
                    d[key] = float(val)
                else:
                    d[key] = val
        else:
            vals = [val.strip() for val in line.replace('-', ' -').split(' ') if val != '']
            if headers is None:
                headers = vals.copy()
                for h in headers:
                    d[h] = []
            else:
                for i, val in enumerate(vals):
                    d[headers[i]].append(float(val))

    # go thru and turn into np.arrays now
    for h in headers:
        d[h] = np.array(d[h])

        if 'REx10^' in h:  # detect and convert RE order of magnitude
            old_res = d.pop(h)
            exponent = int(h.strip('REx10^'))
            d['RE'] = old_res * 10 ** exponent

    return d


def read_xrotor_cput_file():
    return


def read_2col_file(fpath: str, del_after: bool = True):
    with open(fpath, 'r') as f:
        txt = f.read().strip()
    col1 = []
    col2 = []
    lines = txt.split('\n')
    for line in lines:
        val1, val2 = line.strip().split(' ', 1)
        col1.append(float(val1.strip()))
        col2.append(float(val2.strip()))

    if del_after:
        os.remove(fpath)

    return np.array(col1), np.array(col2)


def read_profile_xyz(fpath: str):
    with open(fpath, 'r') as f:
        txt = f.read().strip()
    lines = txt.split('\n')[1:]

    xpts, ypts, zpts = [], [], []
    for line in lines:
        xval, yval, zval = [float(val) for val in line.split(', ')]
        xpts.append(xval)
        ypts.append(yval)
        zpts.append(zval)

    return np.vstack([np.array(xpts), np.array(ypts), np.array(zpts)])


# =============== INTERFACING WITH 3RD PARTY PROGRAMS ===============
def calc_re(rho: float = None, vel: float = None, chord: float = None, mu: float = None, rpm: float = None,
            radius: float = None):
    if all([i is not None for i in [rho, vel, chord, mu]]):
        re = rho * vel * chord / mu
    elif all([i is not None for i in [rho, rpm, radius, chord, mu]]):
        re = rho * (rpm / 60 * 2 * np.pi) * radius * chord / mu
    else:
        raise ValueError('PDT ERROR: Must input all of either [rho, vel, chord, mu] or [rho, rpm, radius, chord, mu]')
    return re


def create_radial_stations(prop: Propeller, plot_also: bool = True, verbose: bool = False):
    # get density for Re estimates
    if prop.design_atmo_props['altitude'] == -1:
        rho, nu = 1000, 0.1150e-5
        mu = nu * rho
    else:
        atmo = standard_atmosphere(prop.design_atmo_props['altitude'])# temp, p, rho, sonic_a, mu
        mu, rho = atmo['mu'], atmo['rho']
        nu = mu / rho

    # replace it if it was entered as atmo prop as well
    if 'dens' in prop.design_atmo_props:
        rho = prop.design_atmo_props['dens']

    t = ''
    stations = []
    for idx, xi in enumerate(prop.station_params):  # append station text all together and save
        if verbose:
            Info('Auto-generating section inputs from airfoil database data for section {} ({})...'.
                  format(idx + 1, prop.station_params[xi]))
        fn = prop.station_params[xi]
        foil = Airfoil(name=fn)
        re_est = int(calc_re(rho=rho, rpm=prop.design_rpm, radius=prop.radius * xi, chord=prop.radius / 10, mu=mu))
        st = RadialStation(station_idx=idx, Xisection=xi, foil=foil, re_estimate=re_est, plot=plot_also)
        t += st.generate_txt_params()
        stations.append(st)
    return stations, t


def create_propeller(name: str, nblades: int, radius: float, hub_radius: float, hub_wake_disp_br: float,
                     design_speed: float, design_cl: dict, design_atmo_props: dict, design_vorform: str,
                     station_params: dict = None, design_adv: float = None, design_rpm: float = None,
                     design_thrust: float = None, design_power: float = None, n_radial: int = 50,
                     verbose: bool = False, show_station_fit_plots: bool = True, plot_after: bool = True,
                     tmout: int = 20, hide_windows: bool = True, geo_params: dict = {}):
    # name must be less than 38? characters for XROTOR to be able to save it
    if len(name) > 38:
        raise Error('"name" must be less than 38 characters when creating a propeller, "{}" is too long'.format(name))

    # must input an altitude in atmo props
    if 'altitude' not in design_atmo_props:
        raise Error('You must include "altitude" as an input to create_propeller()')

    # delete if exists already, make save folder, point-cloud folder
    save_folder = os.path.join(get_prop_db(), name)
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.mkdir(save_folder)

    # make a folder for all the profile x, y, z listings
    prof_folder = os.path.join(save_folder, 'blade_profiles')
    os.mkdir(prof_folder)

    # create the Propeller object, create the stations
    prop = Propeller(name=name, nblades=nblades, radius=radius, hub_radius=hub_radius,
                     hub_wake_disp_br=hub_wake_disp_br, design_speed=design_speed, design_cl=design_cl,
                     design_atmo_props=design_atmo_props, design_vorform=design_vorform, design_adv=design_adv,
                     station_params=station_params, geo_params=geo_params, design_rpm=design_rpm,
                     design_thrust=design_thrust, design_power=design_power)
    st_txt = prop.set_stations(plot_also=show_station_fit_plots, verbose=verbose)

    # prep XROTOR commands depending on what station_params were input
    aero_params_fname = os.path.join(get_prop_db(), '{}_temp_section_params.txt'.format(name))
    with open(aero_params_fname, 'w') as f:
        f.write(st_txt)

    # prep XROTOR commands depending on what design_atmo_props were given
    alt = design_atmo_props['altitude']
    atmo_txt = '{}\n\n'.format(alt)

    if 'vsou' in design_atmo_props:
        vsou = design_atmo_props['vsou']
        atmo_txt += 'vsou\n{}\n'.format(vsou)
    if 'dens' in design_atmo_props:
        dens = design_atmo_props['dens']
        atmo_txt += 'dens\n{}\n'.format(dens)
    if 'visc' in design_atmo_props:
        visc = design_atmo_props['visc']
        atmo_txt += 'visc\n{}\n'.format(visc)
    atmo_txt += 'desi'

    # prep XROTOR commands depending on what vortex formulation is given
    if design_vorform is None:
        vorform_txt = 'pot\n'
    elif any([design_vorform.lower() == i for i in ['grad', 'pot', 'vrtx']]):
        vorform_txt = '{}\n'.format(design_vorform.lower())

    # prep XROTOR commands for the advance ratio OR RPM design specification
    if design_adv is not None:
        if design_rpm is not None:
            raise Error('Cannot specify both "design_adv" and "design_rpm", must pick one or the other')
        adv_rpm_txt = '{}'.format(design_adv)
    elif design_rpm is not None:
        if design_adv is not None:
            raise Error('Cannot specify both "design_adv" and "design_rpm", must pick one or the other')
        adv_rpm_txt = '{}\n{}'.format(0, design_rpm)

    # prep XROTOR commands for the thrust OR power design specification
    if design_thrust is not None:
        thr_pow_txt = '{}'.format(design_thrust)
    elif design_power is not None:
        thr_pow_txt = '{}\n{}'.format(0, design_power)

    # prep XROTOR commands depending on what design_cl keys were given
    if len(design_cl) == 1:  # either a constant value was given, or a filepath
        if 'const' in design_cl:
            const_cl = design_cl['const']
            cl_txt = 'cc\n{}\n\n'.format(const_cl)
        elif 'file' in design_cl:
            fpath = design_cl['file']
            cl_txt = 'cr\n{}\n\n'.format(fpath)
        else:
            raise Error('Must give either a "const" CL target (constant), a "root" and "tip" CL target '
                        '(linearly varied), or specify a CL(r/R) .txt file')
    elif len(design_cl) == 2:  # a linear cl was given
        if all([k in design_cl for k in ['root', 'tip']]):
            root_cl = design_cl['root']
            tip_cl = design_cl['tip']
            cl_txt = 'cl\n{}\n{}\n\n'.format(root_cl, tip_cl)
        else:
            raise Error('If only 2 keywords are given in "design_cl", they must be "root" and "tip"')
    else:
        raise Error('"design_cl" input dictionary error, either too many or not enough keys (if a single key'
                    'is given, it must be either "const" or "file", if 2 keys are given they must be "root" '
                    'and "tip"')

    # prep XROTOR commands for saving blade data r/R
    blade_data_keys = ['CH', 'BE', 'GAM', 'CL', 'CD', 'RE', 'EFP', 'Ub', 'VA', 'VT', 'VD', 'VA/V', 'VT/V', 'VD/V',
                       'VAslip', 'VTslip', 'Aslip', 'Ti', 'Pi', 'Tv', 'Pv', 'Ttot', 'Ptot', 'Xw', 'Vw', 'Tw', 'Pw']
    blade_data = {}
    blade_txt = 'plot\n12\n'
    for key in blade_data_keys:
        fname = '{}_out.txt'.format(key.replace('/', '_over_'))
        blade_data[key] = None
        blade_txt += 'o\n{}\nw\n{}\n'.format(key, fname)

    # prep XROTOR commands to save the solved operating point params
    op_fpath = '{}.xrop'.format(name)
    save_op_txt = 'oper\nwrit\n{}\n'.format(op_fpath)

    # prep XROTOR commands for savename
    savename = '{}.xrr'.format(name)
    save_txt = 'save {}\nquit\n'.format(savename)

    # write XROTOR commands to a file and run in a subprocess
    xrotor_cmnd_file = os.path.join(get_prop_db(), 'xrotor_inputs_temp.txt')
    cmnds = ['aero', 'read', '{}\n'.format(aero_params_fname),
             'desi', 'atmo', '{}'.format(atmo_txt), 'form', '{}'.format(vorform_txt), 'N', '{}'.format(n_radial),
             'inpu', '{}'.format(nblades),
             '{}'.format(radius), '{}'.format(hub_radius), '{}'.format(hub_wake_disp_br), '{}'.format(design_speed),
             '{}'.format(adv_rpm_txt), '{}'.format(thr_pow_txt), '0', '{}'.format(cl_txt), '{}'.format(blade_txt), '\n',
             '{}'.format(save_op_txt), '\n\n', '{}'.format(save_txt)]

    with open(xrotor_cmnd_file, 'w') as f:
        f.write('\n'.join(cmnds))

    xrotor_fpath = os.path.join(get_prop_db(), 'xrotor.exe')
    old_cwd = os.getcwd()
    os.chdir(os.path.join(get_prop_db(), name))
    with open(xrotor_cmnd_file, 'r') as f:
        sui = subprocess.STARTUPINFO()
        if hide_windows:
            sui.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        if not verbose:
            Info('Running XROTOR to create new geometry...')
        subprocess.run([xrotor_fpath], startupinfo=sui, stdin=f, #stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                       timeout=tmout)
    os.chdir(old_cwd)
    os.remove(xrotor_cmnd_file)
    os.remove(aero_params_fname)

    # set the prop's blade-data that we just saved in creation
    for key in blade_data.copy():
        fp = os.path.join(get_prop_db(), '{}\\{}_out.txt'.format(name, key.replace('/', '_over_')))
        roR, array = read_2col_file(fpath=fp)
        if 'r/R' not in blade_data:
            blade_data['r/R'] = roR
        blade_data[key] = array
    prop.set_blade_data(blade_dict=blade_data)

    # interpolate the profiles as part of geometry creation
    if 'n_prof_pts' not in geo_params:
        geo_params['n_prof_pts'] = None
    if 'n_profs' not in geo_params:
        geo_params['n_profs'] = 50
    if 'tot_skew' not in geo_params:
        geo_params['tot_skew'] = 0.0

    prop.interp_foil_profiles(**geo_params)  # also saves the profiles

    # save the PDT propeller meta-file, and then read in the operating point dictionary by calling load_from_savefile()
    prop.save_meta_file()
    prop.load_from_savefile()   # reads meta, xrr, xrop, and point clouds

    if not verbose:
        Info('"{}" Geometry Created!'.format(prop.name))
    if plot_after:
        prop.plot_geometry()

    return prop


def write_blade_cl_file(r_pts: list, cl_pts: list, savepath: str = None):
    if savepath is None:
        savepath = 'blade_design_cl.txt'
    with open(savepath, 'w') as f:
        for r, cl in zip(r_pts, cl_pts):
            f.write('   {r: .6f}      {cl: .6f}\n'.format(r=r, cl=cl))


def convert_ps2png(ps_fpath: str, return_pil_img: bool = False, show_in_pyplot: bool = False):
    """
    Function that converts a PostScript (.ps) image file to a .png file.  Uses 3rd party GNU licensed
    GhostScript, which is capable of much more than just .ps to .png conversion.

    :param ps_fpath: the filepath to the PostScript file
    :param return_pil_img: returns a PIL image object if True, by default just returns the PNG filepath
    :param show_in_pyplot: displays the image in a pyplot figure if True, default is False
    :return png_fpath: the filepath to the converted file
    :return pil_img: the PIL image object
    :return fig: the pyplot figure object
    """
    png_fpath = ps_fpath.replace('.ps', '.png')
    gs_cmnds = ['gswin64c.exe', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-sDEVICE=png16m',
                '-dGraphicsAlphaBits=4', '-sOutputFile={}'.format(png_fpath), '{}'.format(ps_fpath)]
    subprocess.run(gs_cmnds, stdout=subprocess.DEVNULL)  # stderr=subprocess.DEVNULL

    if not return_pil_img and not show_in_pyplot:
        return png_fpath

    from PIL import Image
    pil_img = Image.open(png_fpath)

    if return_pil_img:
        return pil_img

    fig = plt.figure(figsize=[9, 11])
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.imshow(pil_img)
    fig.tight_layout()
    return fig


# =============== MODELING / PHYSICS ===============
def standard_atmosphere(altitude_km: float) -> dict:  # standard atmosphere eqs from Jake's college prof.

    alt_m = altitude_km * 1000

    g, R_air, gamma_air = 9.80665, 287.05307, 1.4
    alt1, alt2, alt3 = 11000, 20000, 32000
    temp0, p0, rho0, mu0 = 288.15, 101325, 1.2250, .0000181206
    slope0, slope1, slope2 = -.0065, 1, .001

    temp1 = temp0 + slope0 * alt1
    p1 = p0 * (temp1 / temp0) ** (-g / slope0 / R_air)
    rho1 = rho0 * (temp1 / temp0) ** (-g / slope0 / R_air - 1)

    temp2 = temp1
    p2 = p1 * np.exp(-g / R_air / temp2 * (alt2 - alt1))
    rho2 = rho1 * np.exp(-g / R_air / temp2 * (alt2 - alt1))

    if alt_m < 0:
        raise Error('Cannot input negative values into standard_atmosphere()')
    elif alt_m < alt1:
        temp = temp0 + slope0 * alt_m
        p = p0 * (temp / temp0) ** (-g / slope0 / R_air)
        rho = rho0 * (temp / temp0) ** (-g / slope0 / R_air - 1)
    elif alt_m < alt2:
        temp = temp1
        p = p1 * np.exp(-g / slope1 / R_air / temp * (alt_m - alt1))
        rho = rho1 * np.exp(-g / R_air / temp * (alt_m - alt1))
    elif alt_m < alt3:
        temp = temp2 + slope2 * (alt_m - alt2)
        p = p2 * (temp / temp2) ** (-g / slope2 / R_air)
        rho = rho2 * (temp / temp2) ** (-g / slope2 / R_air - 1)
    else:
        raise Error('Altitude out of range of atmosphere model')

    sonic_a = np.sqrt(gamma_air * R_air * temp)
    mu = mu0 * (temp / temp0) ** 1.5 * ((temp0 + 120) / (temp + 120))

    return {'temp': temp, 'p': p, 'rho': rho, 'sonic_a': sonic_a, 'mu': mu}


def xrotor_drag_model(CL: np.ndarray, CDmin: float, CLCDmin: float, dCDdCL2: float):
    CD = CDmin + dCDdCL2 * (CLCDmin - CL) ** 2
    return CD


def get_xrotor_re_scaling_exp(re: int):
    re_pts = [0, 1e5, 2e5, 8e5, 2e6, 3e6]
    f_pts = [-0.3, -0.5, -0.5, -1.5, -0.2, -0.1]
    return np.interp(x=re, xp=re_pts, fp=f_pts)


def calc_thrust(k_t, rho, rpm, dia):
    n = rpm / 60
    return k_t * rho * n ** 2 * dia ** 4


# ===== GEOMETRY MANIPULATION =====
def generate_3D_profile_points(nondim_xy_coords: np.ndarray, radius: float, axis_shift: float = 0.5,
                               chord_len: float = 1.0, beta_deg: float = 0.0, skew_deg: float = 0.0):
    xpts, ypts = nondim_xy_coords.copy()

    # mirror over y and shift (0, 0) to align to the blade axis
    xpts = -xpts + axis_shift

    # scale the profile by the chord length
    xpts *= chord_len
    ypts *= chord_len

    # rotate profile by beta
    b = np.deg2rad(beta_deg)
    xp = xpts * np.cos(-b) + ypts * np.sin(-b)
    yp = ypts * np.cos(-b) - xpts * np.sin(-b)

    # create array of radii points
    rp = np.ones(len(xp)) * radius

    # rotate by the local skew angle
    sk = np.deg2rad(skew_deg)
    xs = rp * np.cos(sk) + xp * np.sin(sk)
    ys = xp * np.cos(sk) - rp * np.sin(sk)
    zs = yp

    return np.vstack([xs, ys, zs])
