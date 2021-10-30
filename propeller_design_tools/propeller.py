import os
from propeller_design_tools import funcs
from propeller_design_tools.user_io import Info, Error
from propeller_design_tools.user_settings import get_setting
from propeller_design_tools.airfoil import Airfoil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits import mplot3d
import numpy as np
from stl import mesh


class Propeller(object):

    creation_attrs = {'nblades': int, 'radius': float, 'hub_radius': float, 'hub_wake_disp_br': float,
                      'design_speed_mps': float, 'design_adv': float, 'design_rpm': float, 'design_thrust': float,
                      'design_power': float, 'design_cl': dict, 'design_atmo_props': dict, 'design_vorform': str,
                      'station_params': dict, 'geo_params': dict}
    saveload_attrs = {**creation_attrs, **{'name': str, 'meta_file': str, 'xrr_file': str, 'xrop_file': str,
                                           'blade_data': dict, 'blade_xyz_profiles': dict}}

    def __init__(self, name, **kwargs):
        # name is always given
        self.name = name.replace('.txt', '')

        # set the save folder attr
        self.save_folder = os.path.join(get_setting('propeller_database'), self.name)
        self.meta_file = os.path.join(self.save_folder, '{}.meta'.format(self.name))
        self.xrr_file = os.path.join(self.save_folder, '{}.xrr'.format(self.name))
        self.xrop_file = os.path.join(self.save_folder, '{}.xrop'.format(self.name))
        self.bld_prof_folder = os.path.join(self.save_folder, 'blade_profiles')
        self.stl_mesh = None
        self.stl_fpath = os.path.join(self.save_folder, '{}.stl'.format(self.name))

        # initialize all attrs to None for script auto-completion detection
        self.nblades, self.radius, self.hub_radius, self.hub_wake_disp_br, self.design_speed_mps, self.design_adv, \
        self.design_rpm, self.design_thrust, self.design_power, self.design_cl, self.design_atmo_props, \
        self.design_vorform, self.station_params, self.geo_params, self.xrotor_d, self.xrotor_op_dict, \
        self.blade_data, self.blade_xyz_profiles = [None] * 18

        # if no kwargs were given and there's a meta_file, load it
        if len(kwargs) == 0:
            if os.path.exists(self.meta_file):
                self.load_from_savefile()
            else:
                raise FileNotFoundError('Could not find file {}'.format(self.meta_file))
        else:  # cycle thru kwargs and set if they're valid, if not ignore them
            for key, val in kwargs.items():
                if key in self.creation_attrs:
                    setattr(self, key, val)
                else:
                    raise Error('Unknown KWARG input "{}"'.format(key))

    @property
    def design_rho(self):
        if 'dens' in self.design_atmo_props:
            return self.design_atmo_props['dens']

    @design_rho.setter
    def design_rho(self, value):
        self.design_atmo_props['dens'] = value

    @property
    def disk_area_m_sqrd(self):
        if self.radius is not None and self.hub_radius is not None:
            return np.pi * (self.radius ** 2 - self.hub_radius ** 2)

    @property
    def ideal_eff(self):
        req_attrs = [self.design_thrust, self.design_rho, self.disk_area_m_sqrd, self.design_speed_mps]
        if all([attr is not None for attr in req_attrs]):
            return funcs.calc_ideal_eff(*req_attrs)

    @property
    def disk_loading(self):
        if self.design_thrust is not None and self.disk_area_m_sqrd is not None:
            return self.design_thrust / self.disk_area_m_sqrd

    def read_pdt_metafile(self):
        # read in the PDT meta-file (in the root propeller database) and set Propeller attrs
        with open(self.meta_file, 'r') as f:
            txt = f.read().strip()
        lines = txt.split('\n')

        blade_data = {}
        point_cloud = {}

        for line in lines:
            line_attr, line_val = [i.strip() for i in line.split(':', 1)]
            if 'blade_data_' in line_attr:
                blade_data[line_attr.split('blade_data_', 1)[1]] = np.array([float(v) for v in line_val.strip().split(', ')])
            if 'point_cloud_' in line_attr:
                point_cloud[line_attr.split('point_cloud_', 1)[1]] = np.array([float(v) for v in line_val.strip().split(', ')])
            for attr, tipe in self.saveload_attrs.items():
                if line_attr == attr:
                    val = line.split(': ', 1)[1]
                    if val == 'None':
                        val = None
                    else:
                        if tipe is int:
                            val = int(val)
                        elif tipe is float:
                            val = float(val)
                        elif tipe is dict:
                            items = [item.split(': ') for item in
                                     val.replace('{', '').replace('}', '').replace("'", '').split(', ')]
                            # check for keys that are floats or ints
                            keys_are_nums = [itm[0].replace('.', '').isnumeric() for itm in items]
                            if all(keys_are_nums):
                                for i, itm in enumerate(items):
                                    new_itm = [float(itm[0]), itm[1]]
                                    items[i] = new_itm
                            val = {}
                            for key, entry in items:
                                if 'none' == entry.lower():
                                    entry = None
                                try:    # try to convert to float
                                    val[key] = float(entry)
                                except:     # otherwise store as string
                                    val[key] = entry
                        elif tipe is list:
                            val = []
                    setattr(self, attr, val)
        self.set_blade_data(blade_dict=blade_data)
        self.set_bld_profiles(point_dict=point_cloud)

    def read_xrotor_restart(self):
        with open(self.xrr_file, 'r') as f:
            txt = f.read().strip()
        lines = txt.split('\n')

        def read_line_pair(kw_idx):
            keywords = lines[kw_idx].strip('!').split()
            values = lines[kw_idx + 1].split()
            return dict(zip(keywords, values))

        def read_xi_sect(xi_idx):
            xi_d = {}
            for lni in [xi_idx, xi_idx + 2, xi_idx + 4, xi_idx + 6, xi_idx + 8]:
                for k, v in read_line_pair(lni).items():
                    xi_d[k] = float(v)
            return xi_d

        def read_geo_stations(header_idx):
            geo_d = {}
            headers = lines[header_idx].strip('!').split()
            for h in headers:
                geo_d[h] = []
            for idx in range(header_idx + 1, len(lines)):
                vals = lines[idx].split()
                if len(vals) == len(headers):
                    for i, val in enumerate(vals):
                        geo_d[headers[i]].append(float(val))
            return geo_d

        d = {}
        design_param_lines = [2, 4, 6, 8]
        for ln_idx in design_param_lines:
            for k, v in read_line_pair(ln_idx).items():
                d[k] = float(v)

        xi_lines = [i * 10 for i in range(1, int(d['Naero']) + 1)]   # works out that each xi section is 10 lines, and 1st one on line 10 always
        for sect_idx, ln_idx in enumerate(xi_lines):
            xi_d = read_xi_sect(ln_idx)
            d['Xisection_{}'.format(sect_idx)] = xi_d

        for k, v in read_line_pair(kw_idx=xi_lines[-1] + 10).items():
            d[k] = v

        for k, v in read_line_pair(kw_idx=xi_lines[-1] + 12).items():
            d[k] = int(v)

        for k, v in read_geo_stations(header_idx=xi_lines[-1] + 14).items():
            d[k] = v

        # turn all lists into numpy arrays before returning
        for key, val in d.items():
            if isinstance(val, list):
                d[key] = np.array(val)

        return d

    def load_from_savefile(self):
        # 1st set attrs from the PDT metafile
        self.read_pdt_metafile()

        # set the stations... why did I do it this way??
        self.set_stations(plot_also=False, verbose=True)

        # then read in the XROTOR restart file (in the xrotor_geometry_files)
        self.xrotor_d = self.read_xrotor_restart()

        # then read the operating point output file (in xrotor_op_files)
        self.xrotor_op_dict = funcs.read_xrotor_op_file(fpath=self.xrop_file)

        # and finally read in the point cloud files
        self.blade_xyz_profiles = {}
        fnames = funcs.search_files(folder=self.bld_prof_folder)
        for fname in fnames:
            prof_num = int(fname.replace('profile_', '').replace('.txt', ''))
            xyz_prof = funcs.read_profile_xyz(fpath=os.path.join(self.bld_prof_folder, fname))
            self.blade_xyz_profiles[prof_num] = xyz_prof

        return

    def set_stations(self, plot_also: bool = True, verbose: bool = False):
        self.stations, txt = funcs.create_radial_stations(prop=self, plot_also=plot_also,
                                                        verbose=verbose)
        return txt

    def set_blade_data(self, blade_dict: dict):
        self.blade_data = blade_dict
        return

    def set_bld_profiles(self, point_dict: dict):
        pc = {}
        for key, val in point_dict.items():
            fkey, axkey = key.strip(')').split('(', 1)
            r = float(fkey)
            if r not in pc:
                pc[r] = {}
            pc[r][axkey] = val

        for key, val in pc.items():
            pc[key] = np.vstack([val[ak] for ak in ['x', 'y']])
        self.blade_xyz_profiles = pc
        return

    def save_meta_file(self):
        attrs_2_ignore = ['blade_xyz_profiles']

        if os.path.exists(self.meta_file):
            os.remove(self.meta_file)

        with open(self.meta_file, 'w') as f:
            for attr in [a for a in self.saveload_attrs if a not in attrs_2_ignore]:
                if attr == 'blade_data':
                    for key, val in self.blade_data.items():
                        val = ', '.join([str(i) for i in val])
                        f.write('{}: {}\n'.format('blade_data_{}'.format(key), val))
                else:
                    f.write('{}: {}\n'.format(attr, getattr(self, attr)))

    def get_blade_le_te(self, rotate_deg: float = 0.0, axis_shift: float = 0.25):
        radii = self.radius * np.array(self.xrotor_d['r/R'])
        chords = self.radius * np.array(self.xrotor_d['C/R'])
        betas = np.array(self.xrotor_d['Beta0deg'])
        le_pts = []
        te_pts = []
        ang = np.deg2rad(rotate_deg)
        for radius, chord, beta in zip(radii, chords, betas):
            chord_proj = chord * np.cos(np.deg2rad(beta))
            dz = chord * np.sin(np.deg2rad(beta))
            x_center = radius * np.cos(ang)
            y_center = radius * np.sin(ang)
            x_le = x_center - np.sin(ang) * axis_shift * chord_proj
            x_te = x_center + np.sin(ang) * (1 - axis_shift) * chord_proj
            y_le = y_center + np.cos(ang) * axis_shift * chord_proj
            y_te = y_center - np.cos(ang) * (1 - axis_shift) * chord_proj
            le_pts.append([x_le, y_le, dz * axis_shift])
            te_pts.append([x_te, y_te, -dz * (1 - axis_shift)])
        return le_pts, te_pts

    def interp_foil_profiles(self, n_prof_pts: int = None, n_profs: int = 50, tot_skew: float = 0.0):

        assert len(self.stations) > 0

        if len(self.stations) != 1:
            raise Error('> 1 profile interpolation not yet implemented')

        if tot_skew != 0.0:
            Info('Blade "skew" is not implemented in XROTOR, and therefore not reflected in XROTOR results.\n'
                 '  > Skew effects are considered negligible for PDT purposes for small enough skew angles.')

        # clear out the existing xyz profiles
        funcs.delete_files_from_folder(self.bld_prof_folder)

        station = self.stations[0]
        nondim_coords = station.foil.get_coords(n_interp=n_prof_pts)

        self.blade_xyz_profiles = {}
        for i, roR in enumerate(np.linspace(self.blade_data['r/R'][0], self.blade_data['r/R'][-1], n_profs)):
            chord = np.interp(x=roR, xp=self.blade_data['r/R'], fp=self.blade_data['CH']) * self.radius
            beta = np.rad2deg(np.interp(x=roR, xp=self.blade_data['r/R'], fp=self.blade_data['BE']))
            skew = tot_skew * roR
            r = roR * self.radius

            prof_xyz = funcs.generate_3D_profile_points(nondim_xy_coords=nondim_coords, radius=r, axis_shift=0.25,
                                                        chord_len=chord, beta_deg=beta, skew_deg=skew)
            self.blade_xyz_profiles[i] = prof_xyz

        # now save them all for loading later
        for key, val in self.blade_xyz_profiles.items():
            savepath = os.path.join(self.bld_prof_folder, 'profile_{}.txt'.format(key))
            xpts, ypts, zpts = val
            with open(savepath, 'w') as f:
                f.write('x, y, z\n')
                for xp, yp, zp in zip(xpts, ypts, zpts):
                    f.write('{:.6f}, {:.6f}, {:.6f}\n'.format(xp, yp, zp))

    def plot_geometry(self, LE: bool = True, TE: bool = True, chords_betas: bool = True, hub: bool = True,
                      input_stations: bool = True, interp_profiles: bool = True, savefig: bool = False):
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(nrows=10, ncols=5, figure=fig)
        ax3d = fig.add_subplot(gs[0:7, 0:2], projection='3d')
        txt_ax = fig.add_subplot(gs[7:10, 0:2])
        radial_axes = {'': None, 'c/R': None, 'beta(deg)': None, 'CL': None, 'CD': None,
                       'thrust_eff': None, 'RE': None, 'Mach': None, 'effi': None, 'effp': None,
                       'GAM': None, 'Ttot': None, 'Ptot': None, 'VA/V': None, 'VT/V': None}
        for i, p in enumerate(radial_axes):
            row = i % 5
            col = int(i / 5) + 2
            if col == 2:
                ax = fig.add_subplot(gs[2 * row:2 * row + 2, col])
            else:
                ax = fig.add_subplot(gs[2 * row:2 * row + 2, col])
            radial_axes[p] = ax
            ax.grid(True)
            ax.set_ylabel(p)
            if row == 4:
                ax.set_xlabel('r/R')
            if p == '':
                ax.set_visible(False)

        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')

        title_txt = 'Propeller Geometry - {}'.format(self.name)
        ax3d.set_title(title_txt)

        def do_ax3d():
            blades = np.arange(self.xrotor_d['Nblds'])
            angles = 360 / self.xrotor_d['Nblds'] * blades

            # plot le and te lines
            if LE:
                for ang in angles:
                    le_pts, te_pts = self.get_blade_le_te(rotate_deg=ang)
                    le_line, = ax3d.plot3D(xs=[pt[0] for pt in le_pts], ys=[pt[1] for pt in le_pts],
                                           zs=[pt[2] for pt in le_pts], c='k', lw=2)
            else:
                le_line = None

            if TE:
                for ang in angles:
                    le_pts, te_pts = self.get_blade_le_te(rotate_deg=ang)
                    te_line, = ax3d.plot3D(xs=[pt[0] for pt in te_pts], ys=[pt[1] for pt in te_pts],
                                           zs=[pt[2] for pt in te_pts], c='k', ls='-.', lw=2)
            else:
                te_line = None

            # plot stations
            if chords_betas:
                for ang in angles:
                    le_pts, te_pts = self.get_blade_le_te(rotate_deg=ang)
                    for le_pt, te_pt in zip(le_pts, te_pts):
                        station_line, = ax3d.plot3D(xs=[le_pt[0], te_pt[0]], ys=[le_pt[1], te_pt[1]], zs=[le_pt[2], te_pt[2]],
                                                    c='rosybrown', lw=1, ls='--')
            else:
                station_line = None

            # plot station_params
            if input_stations:
                radii = self.xrotor_d['r/R'] * self.radius
                chords = self.xrotor_d['C/R'] * self.radius
                betas = self.xrotor_d['Beta0deg'].copy()
                for roR, foil_name in self.station_params.items():
                    # station dimensionalized parameters
                    r = roR * self.radius
                    ch = np.interp(r, radii, chords)
                    beta = np.interp(r, radii, betas)
                    sk = self.geo_params['tot_skew'] * roR

                    # load the foil, shift, flip, and dimensionalize coordinates
                    foil = Airfoil(foil_name, verbose=False)
                    xc, yc, zc = funcs.generate_3D_profile_points(nondim_xy_coords=foil.get_coords(), radius=r,
                                                                  axis_shift=0.25, chord_len=ch, beta_deg=beta,
                                                                  skew_deg=sk)
                    foils_line, = ax3d.plot3D(xs=xc, ys=yc, zs=zc, c='red', alpha=0.7, lw=1)
            else:
                foils_line = None

            # plot interpolated profiles
            if interp_profiles:
                for prof_num, prof_xyz in self.blade_xyz_profiles.items():
                    xc, yc, zc = prof_xyz
                    prof_line, = ax3d.plot3D(xs=xc, ys=yc, zs=zc, c='maroon', lw=1, alpha=0.7)
            else:
                prof_line = None

            # plot hub
            if hub:
                hub_thickness = abs(max([pt[2] for pt in le_pts]) - min([pt[2] for pt in te_pts]))
                theta = np.linspace(0, np.pi * 2, 50)
                hub_x = np.cos(theta) * self.hub_radius
                hub_y = np.sin(theta) * self.hub_radius
                top_zs = np.ones(len(hub_x)) * hub_thickness / 2
                bot_zs = -np.ones(len(hub_x)) * hub_thickness / 2
                hub_line, = ax3d.plot3D(xs=hub_x, ys=hub_y, zs=top_zs, c='gray', lw=2)
                hub_line, = ax3d.plot3D(xs=hub_x, ys=hub_y, zs=bot_zs, c='gray', lw=2)
            else:
                hub_line = None

            # set square axes and finish up formatting stuff
            lim = (-self.radius * 0.65, self.radius * 0.65)
            ax3d.set_xlim(lim)
            ax3d.set_ylim(lim)
            ax3d.set_zlim(lim)
            leg_handles = [le_line, te_line, hub_line, foils_line, station_line, prof_line]
            leg_labels = ['L.E.', 'T.E.', 'Hub', 'Input Stations', 'XROTOR Stations', 'Interpolated Geom.']

            leg_labels = [leg_labels[n] for n in range(len(leg_handles)) if leg_handles[n] is not None]
            leg_handles = [leg_handles[n] for n in range(len(leg_handles)) if leg_handles[n] is not None]
            ax3d.legend(leg_handles, leg_labels, loc='upper left', bbox_to_anchor=(1.05, 1.0))

        def do_txt_ax():
            line_num = 0
            txt = ''
            with open(self.xrop_file, 'r') as f:
                while line_num < 16:
                    txt += f.readline()
                    line_num += 1
            txt_ax.text(x=0.0, y=0.5, s=txt, ha='left', va='center', fontfamily='consolas')
            txt_ax.axis('off')

        def do_radial_axes():
            xdata = self.xrotor_op_dict['r/R']
            for ylbl, ax in radial_axes.items():
                if ylbl in self.blade_data:
                    ax.plot(xdata, self.blade_data[ylbl], marker='*', markersize=4)
                else:
                    if ylbl in self.xrotor_op_dict:
                        ax.plot(xdata, self.xrotor_op_dict[ylbl], marker='o', markersize=3)

        def do_thrust_eff_ax():
            ax = radial_axes['thrust_eff']
            ax.set_ylabel('')
            ax.grid(False)
            thrust_eff = self.xrotor_op_dict['thrust(N)'] / self.xrotor_op_dict['power(W)']
            txt1 = 'Thrust Efficiency:\n\n\n\n\nNewtons / Watt'
            ax.text(x=0.5, y=0.5, s=txt1, ha='center', va='center')
            txt2 = '{:.3f}'.format(thrust_eff)
            ax.text(x=0.5, y=0.5, s=txt2, ha='center', va='center', fontsize=12, fontweight='bold')
            ax.axis('off')

            # disk loading metric
            ax.text(x=-0.3, y=0.5, s='Disk Loading:\n\n\n\n\nNewtons / Meter^2', ha='center', va='center')
            ax.text(x=-0.3, y=0.5, s='{:.3f}'.format(self.disk_loading), ha='center', va='center',
                    fontsize=12, fontweight='bold')

        do_ax3d()
        do_txt_ax()
        do_radial_axes()
        do_thrust_eff_ax()
        fig.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.94, wspace=0.35, hspace=0.5)

        if savefig:
            savepath = os.path.join(os.getcwd(), '{}.png'.format(ax3d.get_title()))
            fig.savefig(savepath)
            Info('Saved PNG to "{}"'.format(savepath))

    def generate_stl_geometry(self, plot_after: bool = True, verbose: bool = True):
        n_prof = len(self.blade_xyz_profiles)
        n_pts = np.max(np.shape(self.blade_xyz_profiles[0]))
        n_tri = n_pts * 2 * n_prof

        mdata = np.zeros(n_tri, dtype=mesh.Mesh.dtype)

        tri_idx = 0
        for k in range(n_prof - 1):
            xyz_prof = self.blade_xyz_profiles[k]
            nxt_prof = self.blade_xyz_profiles[k + 1]
            for i in range(n_pts - 1):  # right hand rule to get normal direction correct
                a = xyz_prof[:, i]  # a is a point-coordinate in (x, y, z) format
                b = nxt_prof[:, i]  # same for b-f
                c = nxt_prof[:, i + 1]
                d = a.copy()
                e = c.copy()
                f = xyz_prof[:, i + 1]
                mdata['vectors'][tri_idx] = np.array([a, b, c])      # populate the array of triangle vectors
                mdata['vectors'][tri_idx + 1] = np.array([d, e, f])  # going in order, 2 triangles per iteration
                tri_idx += 2

        m = mesh.Mesh(mdata)

        if os.path.exists(self.stl_fpath):
            os.remove(self.stl_fpath)
        m.save(filename=self.stl_fpath)

        self.stl_mesh = mesh.Mesh.from_file(self.stl_fpath)
        if verbose:
            Info('Saved STL file and reloaded into propeller object: "{}"'.format(self.stl_fpath))

        if plot_after:
            self.plot_stl_mesh()

    def plot_stl_mesh(self):
        fig = plt.figure(figsize=(10, 8))
        ax3d = fig.add_subplot(projection='3d')
        stl_fname = os.path.split(self.stl_fpath)[-1]
        ax3d.set_title(stl_fname)
        ax3d.add_collection3d(mplot3d.art3d.Poly3DCollection(self.stl_mesh.vectors))
        scale = self.stl_mesh.points.flatten()
        ax3d.auto_scale_xyz(scale, scale, scale)

    def plot_ideal_eff(self):
        Info('"{}" ideal efficiency: {:.1f}%'.format(self.name, self.ideal_eff))
        return
