# import os
# import funcs
# import matplotlib.pyplot as plt
# import numpy as np
# # from stl import mesh
# # import time
# # from scipy.optimize import curve_fit
# # from scipy.interpolate import griddata, interp1d
#
#
# class Airfoil(object):
#     def __init__(self, name: str, verbose: bool = True):
#         name_in = name
#         if '.' in name_in:
#             name, ext = os.path.splitext(name_in)
#             if not any([ex in ext.lower() for ex in ['.dat', '.txt']]):
#                 raise ValueError('PDT ERROR: only files with either ".dat" or ".txt" extensions may be used as airfoil '
#                                  'coordinate files ({})'.format(name_in))
#             else:
#                 self.name = name
#         else:
#             self.name = name_in
#         fname = PDT.search_files(folder=PDT.get_airfoil_db_folder(), search_strs=[self.name])[0]
#         self.coord_fpath = os.path.join(PDT.get_airfoil_db_folder(), fname)
#         name, x_coords, y_coords = PDT.read_airfoil_coordinate_file(fpath=self.coord_fpath, verbose=verbose)
#         self.filename = os.path.basename(self.coord_fpath)
#         self.x_coords = np.array(x_coords)
#         self.y_coords = np.array(y_coords)
#
#         self.xfoil_coord_fpath = self.write_xfoil_coord_file()
#         self.polar_data = {}      # dictionary of dictionaries, keys are floats of Re
#         self.rectified_polar_data = {}
#
#         if os.path.exists(self.get_database_savepath()):
#             self.load_polar_data()
#
#     def get_database_savepath(self):
#         database_folder = os.path.join(PDT.get_airfoil_db_folder(), 'polar_database')
#         savename, _ = os.path.splitext(os.path.basename(self.coord_fpath))
#         savepath = os.path.join('{}'.format(database_folder), '{}_polar_data.txt'.format(savename))
#         return savepath
#
#     def write_xfoil_coord_file(self):
#         xfoil_folder = os.path.join(PDT.get_airfoil_db_folder(), 'for_xfoil')
#         if not os.path.exists(xfoil_folder):
#             os.mkdir(xfoil_folder)
#
#         savename, _ = os.path.splitext(os.path.basename(self.coord_fpath))
#         savepath = os.path.join(xfoil_folder, '{}.txt'.format(savename))
#         with open(savepath, 'w') as f:
#             f.write('{}\n\n'.format(self.name))
#             for coord in zip(self.x_coords, self.y_coords):
#                 f.write('{x:.7f} {y:.7f}\n'.format(x=coord[0], y=coord[1]))
#         return savepath
#
#     def plot_geometry(self):
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         ax.plot(self.x_coords, self.y_coords)
#         ax.set_aspect('equal')
#         ax.grid(True)
#         left, right = ax.get_xlim()
#         ax.set_ylim((left - right / 2), -(left - right / 2))
#         ax.set_title(self.filename)
#
#     def alpha_auto_range(self, re: int, ncrit: int, mach: float, suppress_output: bool = True, hide_windows: bool = True):
#         if not suppress_output:
#             PDT.PDT_Info('Detecting range for alpha sweep...')
#
#         def find_alpha_CL0():   # function to find zero lift aoa (A0deg)
#             ts = time.time()
#             PDT.run_xfoil(foil_fpath=self.xfoil_coord_fpath, re=re, cl=[0], ncrit=ncrit, mach=mach,
#                           hide_windows=hide_windows)
#             xout = PDT.read_xfoil_pacc_file(delete_after=True)
#             if xout is not None:    # just return if xfoil converged 1st try
#                 return xout['alpha'][0]
#
#             # otherwise have to get upper and lower points and linearly interpolate
#             cl_lim = 0.02
#             clinc = 0.002
#             cl = 0
#             while xout is None:  # get an upper data point
#                 cl += clinc
#                 PDT.run_xfoil(foil_fpath=self.xfoil_coord_fpath, re=re, cl=[cl], ncrit=ncrit, mach=mach,
#                               keypress_iternum=10, hide_windows=hide_windows)
#                 xout = PDT.read_xfoil_pacc_file(delete_after=True)
#                 if xout is None and cl == cl_lim:
#                     err_txt = 'Unable to find alpha @ CL0 (foil={}, re={:.0f})'.format(self.name, re)
#                     PDT.PDT_Error(InterruptedError, err_txt)
#             a_up = xout['alpha'][0]
#             cl_up = xout['CL'][0]
#
#             cl = 0
#             xout = None
#             while xout is None:  # get a lower data point
#                 cl -= clinc
#                 PDT.run_xfoil(foil_fpath=self.xfoil_coord_fpath, re=re, cl=[cl], ncrit=ncrit, mach=mach,
#                               keypress_iternum=10, hide_windows=hide_windows)
#                 xout = PDT.read_xfoil_pacc_file(delete_after=True)
#                 if xout is None and cl == -cl_lim:
#                     PDT.PDT_Error(InterruptedError, 'Unable to find alpha @ CL0 (foil={}, re={:.0f})'.format(self.name, re))
#
#             a_low = xout['alpha'][0]
#             cl_low = xout['CL'][0]
#
#             da_dcl = (a_up - a_low) / (cl_up - cl_low)
#             dcl = 0 - cl_low
#             A0deg = a_low + dcl * da_dcl
#             return A0deg
#
#         def find_stall_angle(start_a: float = 5, ainc: float = .5, max_a: float = 25, press_iter: int = 10,
#                              dclda_threshold: float = 0.0):   # function to find stall angle
#             this_a = start_a - ainc         # initializing
#             this_cl = 0.0                   # initializing
#             aa = []                         # initializing
#             cl = []                         # initializing
#
#             while this_a < max_a:
#                 last_a = this_a * 1
#                 last_cl = this_cl * 1
#                 this_a += ainc
#                 PDT.run_xfoil(foil_fpath=self.xfoil_coord_fpath, re=re, alpha=[this_a], ncrit=ncrit, mach=mach,
#                               keypress_iternum=press_iter, hide_windows=hide_windows)
#                 xout = PDT.read_xfoil_pacc_file(delete_after=True)
#                 if xout is None:    # when no data was returned (didn't converge)
#                     continue
#                 else:   # when data was returned (converged)
#                     this_cl = xout['CL'][0]  # CL value that was returned
#                     if len(aa) > 0:  # if it's not the first iteration, check for criteria
#                         dcl_da = (this_cl - last_cl) / (this_a - last_a)
#                         if dcl_da < dclda_threshold:
#                             return this_a
#                     aa.append(this_a)
#                     cl.append(this_cl)
#
#             if len(aa) > 0:     # made it to a_max, but converged at least once
#                 return aa[-1] + 2
#             else:   # made it all the way to max_a without converging
#                 trace_txt = 'pdt_classes.Airfoil.alpha_auto_range.find_stall_angle()'
#                 err_txt = 'Unable to find stall angle, max_a reached (foil={}, re={:.0f})'.format(self.name, re)
#
#                 info_d = {'PDT func/method': trace_txt, 'Error Text': err_txt, 'dclda_threshold':
#                     dclda_threshold, 'start_a': start_a, 'ainc': ainc, 'max_a': max_a}
#                 errplot_kwargs = {'x': aa, 'y': cl, 'xlbl': 'alpha (deg)', 'ylbl': 'CL', 'info_d': info_d}
#                 PDT.PDT_Error(e_type=InterruptedError, error_str=err_txt, **errplot_kwargs)
#
#         zero_lift_aoa = find_alpha_CL0()
#         a_stall = find_stall_angle()
#
#         return np.arange(zero_lift_aoa - 1, a_stall + 2.5, 0.5)
#
#     def calculate_xfoil_polars(self, hide_windows: bool = True, **kwargs):
#         """
#         Used to interface with XFOIL and save database data.
#
#         :param re_list: Reynolds numbers to iterate across
#         :param alpha: Angle of Attack (AoA) to iterate across.  Defaults to None which attempts to sweep across
#                         the range of alpha values that are required to best characterize the foil's aero properties in
#                         XROTOR (from Zero-lift AoA up to AoA @ stall)
#         :param ncrit: Defaults to 9. From the XFOIL user docs about ncrit ->
#                          situation             Ncrit
#                       -----------------        -----
#                       sailplane                12-14
#                       motorglider              11-13
#                       clean wind tunnel        10-12
#                       average wind tunnel        9     <=  standard "e^9 method"
#                       dirty wind tunnel         4-8
#         :param mach: The Mach number of the flow
#         :param save_to_database: Defaults to True, which automatically makes the results save to the database
#         :return:
#         """
#
#         # KWARG conditioning
#         if 're' in kwargs:
#             re_list = kwargs.pop('re')
#             if not isinstance(re_list, list):
#                 re_list = [re_list]
#         else:
#             PDT.PDT_Error(ValueError, 'Must input a value for KWARG "re"')
#
#         if 'alpha' in kwargs:
#             alpha_list = kwargs.pop('alpha')
#             if not isinstance(alpha_list, list):
#                 if alpha_list is not None:
#                     alpha_list = [alpha_list]
#         else:
#             alpha_list = None  # default -> alpha_list=None will trigger alpha-auto range
#
#         if 'ncrit' in kwargs:
#             ncrit_list = kwargs.pop('ncrit')
#             if not isinstance(ncrit_list, list):
#                 ncrit_list = [ncrit_list]
#         else:
#             ncrit_list = [9]  # default
#         for nc in ncrit_list:        # check ncrit in range
#             if nc not in range(4, 15):
#                 PDT.PDT_Error(ValueError, 'Parameter ncrit must be an integer 4-14 (got {})'.format(nc))
#
#         if 'mach' in kwargs:
#             mach_list = kwargs.pop('mach')
#             if not isinstance(mach_list, list):
#                 mach_list = [mach_list]
#         else:
#             mach_list = [0.0]  # default
#
#         if 'save_to_database' in kwargs:
#             save_to_database = kwargs.pop('save_to_database')
#         else:
#             save_to_database = True  # default
#
#         if 'suppress_output' in kwargs:
#             suppress_output = kwargs.pop('suppress_output')
#         else:
#             suppress_output = True  # default
#
#         alpha_in = alpha_list
#         total_count = len(re_list) * len(mach_list) * len(ncrit_list)
#         count = 0
#         for ncrit in ncrit_list:
#             for mach in mach_list:
#                 for re in re_list:
#                     count += 1
#                     re = int(re)
#                     if alpha_in is None:
#                         alpha_list = self.alpha_auto_range(re=re, ncrit=ncrit, mach=mach,
#                                                            suppress_output=suppress_output, hide_windows=hide_windows)
#
#                     if not suppress_output:
#                         PDT.PDT_Info('Running polar # {} / {}...'.format(count, total_count))
#                     PDT.run_xfoil(foil_fpath=self.xfoil_coord_fpath, re=re, alpha=alpha_list, ncrit=ncrit,
#                                   mach=mach, hide_windows=hide_windows)
#                     d = PDT.read_xfoil_pacc_file(delete_after=True)
#                     self.polar_data[(re, mach, ncrit)] = d
#
#         # exit method if not saving to database
#         if not save_to_database:
#             return
#
#         # save/append to database, start by making folder if it doesn't already exist
#         database_folder = os.path.join(PDT.get_airfoil_db_folder(), 'polar_database')
#         if not os.path.exists(database_folder):
#             os.mkdir(database_folder)
#
#         # savename will be coordinate file name with _polar_data appended
#         savepath = self.get_database_savepath()
#
#         # append/overwrite data if file exists already, otherwise write data to a new file
#         if os.path.exists(savepath):
#             old_polar_data = PDT.read_polar_data_file(fpath=savepath)
#             merged = PDT.merge_polar_data_dicts(new=self.polar_data.copy(), old=old_polar_data)
#             os.remove(savepath)
#             PDT.save_polar_data_file(polar_data=merged, savepath=savepath, name=self.name)
#         else:
#             PDT.save_polar_data_file(polar_data=self.polar_data, savepath=savepath, name=self.name)
#
#     def get_valid_xfoil_params(self):
#         return ['alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr', 'CL/CD']
#
#     def plot_polar_data(self, x_param: str, y_param: str, re_list: list = None, mach_list: list = None,
#                         ncrit_list: list = None, xlims: tuple = None, ylims: tuple = None, rectified: bool = False,
#                         rect_kwargs: dict = {}, **plot_kwargs):
#         """
#         Method to plot existing polar data.
#
#         :param x_param: the dictionary key for the x-axis data to plot.  Options are alpha, CL, CD, CDp, CM, Top_Xtr,
#             Bot_Xtr -> all directly returned from XFOIL, and CL/CD -> calculated by PDT.
#         :param y_param: the dictionary key for the y-axis data to plot.  Options are alpha, CL, CD, CDp, CM, Top_Xtr,
#             Bot_Xtr -> all directly returned from XFOIL, and CL/CD -> calculated by PDT.
#         :param re_list: list of reynolds numbers to plot. If given None, will plot all existing re.
#         :param mach_list: list of machs to plot.  If given None, will plot all existing mach.
#         :param ncrit_list: list of ncrits to plot.  If given None, will plot all existing ncrit.
#         :param xlims: 2-tuple of (xmin, xmax) for the plot
#         :param ylims: 2-tuple of (ymin, ymax) for the plot
#         :param rectified: bool, whether or not to plot the raw grid data, or rectify it first and plot that data
#         :param rect_kwargs: dictionary, the key-word arguments to pass along to rectify_polar_grids
#
#         :return fig: the pyplot.fig instance of the plot
#         :return ax: the pyplot.axes instance of the plot
#         """
#
#         if x_param not in self.get_valid_xfoil_params() or y_param not in self.get_valid_xfoil_params():
#             raise ValueError('PDT ERROR: "{}" is not a valid polar parameter combo for plotting'.
#                              format('({}, {})'.format(x_param, y_param)))
#
#         pol_data = self.polar_data
#         if rectified:
#             self.rectify_polar_grids(rect_kwargs=rect_kwargs)
#             pol_data = self.rectified_polar_data
#
#         if len(pol_data) == 0:
#             raise ValueError('PDT ERROR: No polar data to plot for "{}"! Must either load data with "load_polar_data()" or use '
#                              '"calculate_xfoil_polars()" first'.format(self.name))
#
#         if 'marker' not in plot_kwargs:
#             plot_kwargs['marker'] = 'o'
#
#         # get the grid of polar lookup keys for the associated polar data
#         re_l, mach_l, ncrit_l = self.get_polar_data_grid()
#
#         # if a polar lookup key isn't given, plot all data across that lookup key by default
#         if re_list is None:
#             re_list = re_l
#         if mach_list is None:
#             mach_list = mach_l
#         if ncrit_list is None:
#             ncrit_list = ncrit_l
#
#         # enforce that the user input valid values
#         re_list = list(sorted([int(r) for r in re_list]))
#         mach_list = list(sorted([float(m) for m in mach_list]))
#         ncrit_list = list(sorted([int(n) for n in ncrit_list]))
#
#         # create figure and axes instance
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111)
#
#         # plot em
#         for re_key in re_list:
#             plot_kwargs['c'] = plt.cm.jet(re_list.index(re_key) / len(re_list))
#             for mach_key in mach_list:
#                 if max(mach_list) == 0:
#                     plot_kwargs['alpha'] = 1
#                 else:
#                     plot_kwargs['alpha'] = mach_key / max(mach_list)
#
#                 for ncrit_key in ncrit_list:
#                     marker_cycle = ['o', 'v', '^', '<', '>', 's', '+', 'd', '2']
#                     plot_kwargs['marker'] = marker_cycle[ncrit_list.index(ncrit_key) % len(marker_cycle)]
#                     if (re_key, mach_key, ncrit_key) in pol_data.keys():
#                         d = pol_data[(re_key, mach_key, ncrit_key)]
#                         ax.plot(d[x_param], d[y_param], label='{}, {}, {}'.format(re_key, mach_key, ncrit_key), **plot_kwargs)
#                     else:
#                         print('Warning: Cannot find polar for {} @ Re = {}, Mach = {}, Ncrit = {}, skipping this one...'
#                               .format(self.name, re_key, mach_key, ncrit_key))
#
#         ax.grid(True)
#         ax.set_title('{}\nrectified={}'.format(self.filename, rectified))
#         ax.set_xlabel(x_param)
#         ax.set_ylabel(y_param)
#         if ylims is not None:
#             ax.set_ylim(ylims)
#         if xlims is not None:
#             ax.set_xlim(xlims)
#         ax.legend(loc='upper left', title='Re, Mach, Ncrit', bbox_to_anchor=(1.01, 1.0))
#         fig.subplots_adjust(left=0.09, right=0.79)
#
#         return fig, ax
#
#     def get_polar_data_grid(self):
#         re_list, mach_list, ncrit_list = [], [], []
#         for key in self.polar_data.keys():
#             re_list.append(key[0])
#             mach_list.append(key[1])
#             ncrit_list.append(key[2])
#         return list(set(re_list)), list(set(mach_list)), list(set(ncrit_list))
#
#     def get_keys_2_interpolate(self):
#         k = self.get_valid_xfoil_params()
#         k.pop(k.index('CL/CD'))
#         # k.pop(k.index('alpha'))
#         return k
#
#     def rectify_polar_grids(self, **kwargs):
#         if not self.rectified_polar_data == {}:
#             return
#
#         if 'interp_kind' in kwargs:
#             interp_kind = kwargs.pop('interp_kind')
#         else:
#             interp_kind = 'linear'
#
#         rect_polar_data = self.polar_data.copy()
#
#         keys2rect = self.get_keys_2_interpolate()
#         if 'alpha' in keys2rect:
#             keys2rect.pop(keys2rect.index('alpha'))
#
#         alpha_mins = [np.min(pol['alpha']) for pol in rect_polar_data.values()]
#         alpha_maxes = [np.max(pol['alpha']) for pol in rect_polar_data.values()]
#
#         alpha_rect = np.linspace(np.min(alpha_mins), np.max(alpha_maxes), 50)
#         for pol in rect_polar_data.values():
#             for key in keys2rect:
#                 ydata = pol[key]
#                 pol_interpolator = interp1d(x=pol['alpha'], y=ydata, kind=interp_kind, fill_value='extrapolate')
#                 pol[key] = pol_interpolator(alpha_rect)
#
#             pol['alpha'] = alpha_rect
#
#         self.rectified_polar_data = rect_polar_data
#
#     def interpolate_polar(self, npts: int, re: int, mach: float, ncrit: int, griddata_kwargs: dict = {}):
#         # check for inside convex hull of known (Re, Mach, nCrit) grid points
#         res, machs, ncrits = self.get_polar_data_grid()
#         if re > max(res) or re < min(res):
#             PDT.PDT_Error(ValueError, 'Cannot interpolate a polar @ re value outside database limits '
#                                       '(re={:.0f})'.format(re))
#         if mach > max(machs) or re < min(machs):
#             PDT.PDT_Error(ValueError, 'Cannot interpolate a polar @ mach value outside database limits '
#                                       '(mach={:.2f})'.format(mach))
#         if ncrit > max(ncrits) or re < min(ncrits):
#             PDT.PDT_Error(ValueError, 'Cannot interpolate a polar @ ncrit value outside database limits '
#                                       '(ncrit={:.0f})'.format(ncrit))
#
#         if 'method' not in griddata_kwargs:
#             griddata_kwargs['method'] = 'linear'
#         if 'rescale' not in griddata_kwargs:
#             griddata_kwargs['rescale'] = True
#
#         # send em right back if their desired polar is one of the grid points
#         if (re, mach, ncrit) in self.polar_data:
#             return self.polar_data[(re, mach, ncrit)].copy()
#
#         # rectify polar grids
#         self.rectify_polar_grids(interp_kind='linear')
#
#         # get all the keys to interpolate, will interpolate across 2D planes of x=alpha, y=key
#         keys2interp = self.get_keys_2_interpolate()
#
#         # flatten values across database Re, mach, ncrit
#         flat_vals = {k: np.array([]) for k in keys2interp}
#         re_pts, mach_pts, ncrit_pts = [], [], []
#         for pol_key, pol in self.rectified_polar_data.items():
#             re_pts.extend([pol_key[0]] * len(pol['alpha']))
#             mach_pts.extend([pol_key[1]] * len(pol['alpha']))
#             ncrit_pts.extend([pol_key[2]] * len(pol['alpha']))
#             for key in keys2interp:
#                 new_vals = np.append(flat_vals[key], pol[key].flatten())
#                 flat_vals[key] = new_vals
#
#         alphas = flat_vals.pop('alpha')
#         keys2interp.pop(keys2interp.index('alpha'))
#         pol_interp = {}
#         alpha_interp = np.linspace(np.min(alphas), np.max(alphas), npts)
#         pol_interp['alpha'] = alpha_interp
#
#         if len(machs) == 1 and len(ncrits) == 1:    # if there's only 1 mach and 1 ncrit datapoint
#             points = np.full(shape=(len(alphas), 2), fill_value=np.nan)
#             points[:, 0] = np.array(re_pts)
#             points[:, -1] = alphas
#
#             pol_interp['re'] = re
#             pol_interp['mach'] = machs[0]
#             pol_interp['ncrit'] = ncrits[0]
#
#             for key in keys2interp:
#                 values = flat_vals[key]
#                 re_interp = np.ones(shape=(len(alpha_interp), 1)) * re
#                 alpha_interp = np.reshape(alpha_interp, (len(alpha_interp), 1))
#                 xi = np.hstack([re_interp, alpha_interp])
#                 pol_interp[key] = griddata(points=points, values=values, xi=xi, **griddata_kwargs)
#
#         elif len(list(set(machs))) == 1:    # so there's data across re and ncrit
#             points = np.full(shape=(len(alphas), 3), fill_value=np.nan)
#             points[:, 0] = np.array(re_pts)
#             points[:, 1] = np.array(ncrit_pts)
#             points[:, -1] = alphas
#
#             pol_interp['re'] = re
#             pol_interp['mach'] = machs[0]
#             pol_interp['ncrit'] = ncrit
#
#             for key in keys2interp:
#                 values = flat_vals[key]
#                 re_interp = np.ones(shape=(len(alpha_interp), 1)) * re
#                 ncrit_interp = np.ones(shape=(len(alpha_interp), 1)) * ncrit
#                 alpha_interp = np.reshape(alpha_interp, (len(alpha_interp), 1))
#                 xi = np.hstack([re_interp, ncrit_interp, alpha_interp])
#                 pol_interp[key] = griddata(points=points, values=values, xi=xi, **griddata_kwargs)
#
#         scrubbed_pol = PDT.scrub_nans(d=pol_interp)
#         return scrubbed_pol
#
#     def load_polar_data(self):
#         savepath = self.get_database_savepath()
#         if os.path.exists(savepath):
#             self.polar_data = PDT.read_polar_data_file(fpath=savepath)
#         else:
#             raise FileNotFoundError('Could not find file: {}'.format(savepath))
#
#     def get_coords(self, n_interp: int = None):
#         if n_interp is None:
#             return np.vstack([self.x_coords, self.y_coords])
#         else:
#             PDT.PDT_Error(NotImplementedError, 'Code to interpolate more profile points is incomplete...')
