import os
from propeller_design_tools import funcs
from propeller_design_tools.airfoil import Airfoil
from propeller_design_tools.user_io import Info, Error
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


class RadialStation(object):

    req_xrotor_attrs = ['A0deg', 'dCLdA', 'dCLdAstall', 'CLmax', 'CLmin', 'CLinc2stall', 'CDmin', 'CLCDmin',
                        'dCDdCL2', 'REref', 'REexp', 'Cmconst', 'Mcrit']

    def __init__(self, station_idx: int = None, foil: Airfoil = None, re_estimate: int = None, momma=None,
                 Xisection: float = None, plot: bool = False, verbose: bool = True, **xrotor_kwargs):

        self.station_idx = station_idx
        self.momma = momma
        self.Xisection = Xisection
        self.re_estimate = re_estimate
        self.foil = foil

        # initialize required attrs as None
        self.A0deg, self.dCLdA, self.dCLdAstall, self.CLmax, self.CLmin, self.CLinc2stall, self.CDmin, self.CLCDmin, \
        self.dCDdCL2, self.REref, self.REexp, self.Cmconst, self.Mcrit = [None] * 13

        # need to give re_estimate
        if re_estimate is None:
            raise Error('Must give an Re estimate at which to interpolate data')

        # kick em out if no airfoil given and all the required inputs were not given
        if foil is None:
            raise Error('Must give an Airfoil() as an input into RadialStation()')
            # for attr in self.req_xrotor_attrs:
            #     if attr not in xrotor_kwargs:
            #         raise Error('Required argument "{}" not given'.format(attr))
            #     else:   # set the required inputs if one was given though
            #         setattr(self, attr, xrotor_kwargs[attr])
        else:   # otherwise initialize all the req_attrs automatically from airfoil's data
            if verbose:
                Info('Initializing RadialStation() from foil "{}" interpolated @ Re={}'
                     .format(self.foil.name, self.re_estimate))
            self.init_from_airfoil(foil=foil, re_estimate=re_estimate, plot_also=plot, verbose=verbose)

    def set_mom_prop(self, momma):
        self.momma = momma

    def set_Xisection(self, Xisection):
        self.Xisection = Xisection

    def init_from_airfoil(self, foil: Airfoil, re_estimate: int, mach_estimate: float = 0, ncrit_estimate: int = 9,
                          plot_also: bool = False, verbose: bool = True):
        # interpolate the given estimates
        self.foil = foil
        pol = foil.interpolate_polar(npts=50, re=re_estimate, mach=mach_estimate, ncrit=ncrit_estimate)
        # solve for fit params, plot if also
        idx_lims = self.calc_xrotor_fit_params(pol=pol, verbose=verbose)
        if plot_also:
            self.plot_xrotor_fit_params(pol=pol, idx_lims=idx_lims)

    def xrotor_CL_model(self, a: float):
        # get distance from input alpha, calc CL based on linear slope assumption
        dA = a - self.A0deg
        CL = self.dCLdA * dA

        # if linear slope assumption accurate, return results
        if self.CLmin <= CL <= self.cl_pre_stall():
            return CL

        # if not linear, detect & recalculate CL
        if self.cl_pre_stall() < CL:  # stall model
            # calc 2nd der.
            Ainc2stall = self.CLinc2stall / self.dCLdA
            a_ref = self.a_pre_stall()
        elif CL < self.CLmin:  # stall model flipped over both axes
            Ainc2stall = -self.CLinc2stall / self.dCLdA
            a_ref = self.a_min()

        dCL2dA2 = (self.dCLdAstall - self.dCLdA) / (2 * Ainc2stall)  # how much to change dCLdA over what alpha gap
        delta_a = a - a_ref  # how far from a_ref?
        delta_CL = 0.5 * delta_a**2 * dCL2dA2     # how much CL to add to linear assumption
        CL = CL + delta_CL  # new CL

        # check if reached dCLdAstall yet
        delta_dCLdA = delta_a * dCL2dA2   # change in slope
        dCLdA = self.dCLdA + delta_dCLdA    # slope at the current point
        if dCLdA < self.dCLdAstall:
            a_pt = a_ref + 2 * Ainc2stall
            cl_pt = self.dCLdA * (a_pt - self.A0deg) + 0.5 * (a_pt - a_ref) ** 2 * dCL2dA2
            CL = cl_pt + (a - a_pt) * self.dCLdAstall

        return CL

    def cl_pre_stall(self):  # CL value at top-most linear point
        return self.CLmax - self.CLinc2stall

    def a_pre_stall(self):  # alpha corresponding to cl_pre_stall()
        return self.A0deg + self.cl_pre_stall() / self.dCLdA

    def a_max(self):  # alpha @ stall
        return self.A0deg + self.CLmax / self.dCLdA

    def a_min(self):
        return self.A0deg + self.CLmin / self.dCLdA

    def a_stall(self):
        aoa = np.linspace(self.A0deg, self.a_max() + 5, 200)
        cl = [self.xrotor_CL_model(a) for a in aoa]
        return aoa[np.argmax(cl)]

    def cl_stall(self):
        return self.xrotor_CL_model(self.a_stall())

    def xrotor_drag_model(self, CL: np.ndarray):
        CD = self.CDmin + self.dCDdCL2 * (self.CLCDmin - CL) ** 2
        return CD

    def calc_xrotor_fit_params(self, pol: dict, verbose: bool = True):
        # ===== Fitting the CL(alpha) Curve =====
        # calculate lift-curve slope in /degrees, and post lift-curve slope
        dcl_da = np.diff(pol['CL']) / np.diff(pol['alpha'])  # assume median value of lift-curve slope
        self.dCLdA = np.nanmedian(dcl_da[np.rad2deg(dcl_da) > 3])  # only consider where slope > 3 /radian
        self.dCLdAstall, b2 = np.polyfit(x=pol['alpha'][-3:], y=pol['CL'][-3:], deg=1)

        # pre-stall
        pre_stall_idx = len(dcl_da) - np.where(np.flipud(np.abs(dcl_da - self.dCLdA) < 0.02))[0][0]
        cl_pre_stall = pol['CL'][pre_stall_idx]
        a_pre_stall = pol['alpha'][pre_stall_idx]

        # alpha @ CL=0, -> re-calculate lift-curve slope to be slope of line thru (A0deg, 0) and (a_pre_stall, cl_pre_stall)
        self.A0deg = a_pre_stall - cl_pre_stall * (1 / self.dCLdA)  # this gives known point (A0deg, 0) on CL(a)
        self.dCLdA = (cl_pre_stall - 0) / (a_pre_stall - self.A0deg)  # this gives slope of line
        b1 = -self.dCLdA * self.A0deg

        # stall is when those two lines cross (un-stalled and post-stalled linear sections)
        a_stall = (b2 - b1) / (self.dCLdA - self.dCLdAstall)
        self.CLmax = (a_stall - self.A0deg) * self.dCLdA

        # minimum idx
        min_idx = np.where(np.abs(dcl_da - self.dCLdA) < 0.02)[0][0]
        if pol['CL'][min_idx] < 0:
            self.CLmin = pol['CL'][min_idx]
        else:
            self.CLmin = 0.0
            min_idx = np.where(pol['CL'] >= 0)[0][0]

        # I think these are the definitions of these at least
        self.CLinc2stall = self.CLmax - cl_pre_stall

        # ===== Fitting the CD(CL) Curve =====
        # get "optimal" fitting parameters using scipy.optimize.curve_fit
        if pol['CD'][min_idx] > pol['CD'][pre_stall_idx]:
            min_idx = np.where(pol['CD'] < pol['CD'][pre_stall_idx])[0][0] - 1
        xs = pol['CL'][min_idx:pre_stall_idx]
        ys = pol['CD'][min_idx:pre_stall_idx]
        popt, pcov = curve_fit(f=funcs.xrotor_drag_model, xdata=xs, ydata=ys, bounds=([0.0, -1.0, 0.0], [0.5, 2.5, .5]))
        self.CDmin, self.CLCDmin, self.dCDdCL2 = popt

        # pitching moment
        self.Cmconst = np.average(pol['CM'][min_idx:pre_stall_idx])
        self.REref = pol['re']
        self.REexp = funcs.get_xrotor_re_scaling_exp(re=pol['re'])
        self.Mcrit = 1

        if verbose:
            Info('Found required XROTOR section inputs ->')
            for attr in self.req_xrotor_attrs:
                val = getattr(self, attr)
                Info('{}: {}'.format(attr, val), indent_level=1)

        return min_idx, pre_stall_idx

    def plot_xrotor_fit_params(self, pol: dict, idx_lims: tuple = None):
        # plotting parameters for all the axes
        xlbls = ['CD', 'alpha', 'alpha']
        ylbls = ['CL', 'CL', 'CM']
        xlims = [(0.0, 0.07), None, None]
        kw = {'marker': '*', 'lw': 4, 'alpha': 0.4, 'markersize': 12, 'linestyle': '--'}

        # create figure and axes, turn grids on
        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(1, 4)
        fig.add_subplot(gs[:, :2])
        fig.add_subplot(gs[:, 2])
        fig.add_subplot(gs[:, 3])
        axes = fig.axes

        sect_txt = 'None' if self.station_idx is None else '{}'.format(self.station_idx + 1)
        fig.suptitle('Blade Section {} @ Xi={}\n{} Interpolated @ Re={}, Mach={}, nCrit={}\nXROTOR Section Parametric Modeling'.
                     format(sect_txt, self.Xisection, self.foil.name, pol['re'], pol['mach'], pol['ncrit']))

        # iterate over axes and set them up, also plot the data to be fitted on each
        for ax, xlbl, ylbl, xlim in zip(axes, xlbls, ylbls, xlims):
            ax.grid(True)
            ax.set_xlabel(xlbl)
            ax.set_ylabel(ylbl)
            ax.plot(pol[xlbl], pol[ylbl], label='Interp. Data', **kw)
            ax.set_xlim(xlim)

        # plot CLmax, CLmin lines
        for ax in axes[:2]:
            ax.axhline(self.CLmax, ls='--', c='k', lw=1)
            ax.axhline(self.CLmin, ls='--', c='k', lw=1)
            ax.axhline(self.CLmax - self.CLinc2stall, ls='--', c='k', lw=1)

        # highlight the datapoints considered in the modeling of the linear section
        if idx_lims is not None:
            for ax in axes:
                line = ax.get_lines()[0]
                xdata, ydata = line.get_xdata(), line.get_ydata()
                ax.plot(xdata[idx_lims[0]:idx_lims[1]], ydata[idx_lims[0]:idx_lims[1]], color='orange',
                        label='Used in fitting', **kw)

        # plot the entire XROTOR CL estimate
        col = 'royalblue'
        limbs = axes[1].get_xlim()
        # xs = np.linspace(limbs[0], limbs[1], 50)
        xs = np.linspace(limbs[0], limbs[1], 50)
        ys = np.array([self.xrotor_CL_model(a=x) for x in xs])
        axes[1].plot(xs, ys, color=col, lw=3, label='XROTOR Fit')

        # plot the XROTOR drag polar estimate in just the region corresponding to linear CL
        ys = np.linspace(self.CLmin, self.cl_pre_stall(), 50)
        xs = self.xrotor_drag_model(CL=ys)
        axes[0].plot(xs, ys, c=col, lw=3, label='XROTOR Fit')

        # plot the entire XROTOR CM estimate
        axes[2].axhline(self.Cmconst, color=col, lw=3, label='XROTOR Fit')

        # display the lift-curve fitting parameters on the plot
        ax1_fs, ax1_lw = 10, 1.8
        # A0deg
        axes[1].annotate(text='A0deg = {:.2f} deg'.format(self.A0deg),
                         xy=(self.A0deg, 0), xytext=(self.A0deg + 3, 0), xycoords='data',
                         arrowprops=dict(width=ax1_lw, facecolor='black'), ha='left', va='center', fontsize=ax1_fs)

        # dCLdA
        xy_idx = idx_lims[0] + int((idx_lims[1] - idx_lims[0]) / 2)
        xy = (pol['alpha'][xy_idx], pol['CL'][xy_idx])
        axes[1].annotate(text='dCLdA\n={:.3f}/deg\n={:.2f}/rad'.format(self.dCLdA, self.dCLdA / np.deg2rad(1)),
                         xy=xy, xytext=(xy[0] + 3, xy[1] - 0.25), xycoords='data',
                         arrowprops=dict(width=ax1_lw, facecolor='black'), ha='left', fontsize=ax1_fs)

        # CLmax
        axes[1].annotate(text='CLmax = {:.2f}'.format(self.CLmax),
                         xy=(self.a_stall(), self.CLmax), xytext=(self.a_stall() - 5, self.CLmax * 1.02), xycoords='data',
                         arrowprops=dict(width=ax1_lw, facecolor='black'), ha='right', va='center', fontsize=ax1_fs)

        # CLinc2stall
        axes[1].annotate(text='', xy=(self.a_pre_stall(), self.cl_pre_stall()), xytext=(self.a_pre_stall(), self.CLmax),
                         xycoords='data', arrowprops=dict(arrowstyle='<->', fc='k', lw=ax1_lw), ha='right',
                         va='center', fontsize=ax1_fs)
        axes[1].annotate(text='CLinc2stall\n= {:.2f}'.format(self.CLinc2stall),
                         xy=(self.a_pre_stall(), (self.cl_pre_stall() + self.CLmax) / 2),
                         xytext=(self.a_pre_stall() - 3, self.cl_pre_stall() * 1.1), xycoords='data',
                         arrowprops=dict(arrowstyle='-', fc='k', lw=ax1_lw), ha='right', va='top', fontsize=ax1_fs)

        # dCLdAstall === the lift-curve slope in the post-stall region
        xc = (axes[1].get_xlim()[1] + self.a_stall()) / 2
        yc = self.xrotor_CL_model(xc)
        axes[1].annotate(
            text='dCLdAstall\n={:.3f}/deg\n={:.2f}/rad'.format(self.dCLdAstall, np.rad2deg(self.dCLdAstall)),
            xy=(xc, yc),
            xytext=(xc - 2, yc - 0.25), xycoords='data',
            arrowprops=dict(width=ax1_lw, facecolor='black'), ha='center', va='top', fontsize=ax1_fs)

        # display the quadratic parameter, coords of control point
        ax0_fs, ax0_lw = 12, 2
        axes[0].annotate(text='(CDmin, CLCDmin) = ({:.4f}, {:.4f})'.format(self.CDmin, self.CLCDmin),
                         xy=(self.CDmin, self.CLCDmin), xytext=(self.CDmin * 1.4, self.CLCDmin - 0.1), xycoords='data',
                         arrowprops=dict(width=ax0_lw, facecolor='black'), ha='left', va='center', fontsize=ax0_fs)

        # dCDdCL2 (quadratic parameter)
        yc = (self.CLCDmin + self.cl_pre_stall()) / 2
        xc = self.xrotor_drag_model(yc)
        axes[0].annotate(text='d(CD) / d(CL**2) = {:.5f}'.format(self.dCDdCL2),
                         xy=(xc, yc), xytext=(xc * 1.4, yc - 0.1), xycoords='data',
                         arrowprops=dict(width=ax0_lw, facecolor='black'), ha='left', fontsize=ax0_fs)

        # CLmin, on this axes[0]
        yc = self.CLmin
        xc = self.xrotor_drag_model(yc)
        axes[0].annotate(text='CLmin = {:.3f}'.format(self.CLmin),
                         xy=(xc, yc), xytext=(xc * 1.4, yc + 0.1), xycoords='data',
                         arrowprops=dict(width=ax0_lw, facecolor='black'), ha='left', va='center', fontsize=ax0_fs)

        # final formatting
        fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.08, wspace=0.35)
        axes[0].set_ylim(axes[1].get_ylim())
        axes[2].legend(loc='upper left', bbox_to_anchor=(-0.6, 1.14))

    def save_xrotor_input_txt(self, prop_name, section_num, Xi):
        savepath = os.path.join(os.getcwd(),
                                '{}_section_params_{}_{}.txt'.format(prop_name, section_num, self.foil.name))
        # txt = self.generate_txt_params(section_num=section_num, Xi=Xi)
        txt = self.generate_txt_params()
        with open(savepath, 'w') as f:
            f.write(txt)

    def generate_txt_params(self):
        if self.station_idx is not None:
            sect_num = self.station_idx + 1
        else:
            sect_num = None
        txt = ''
        txt += '\nSection {}   r/R = {:.3f}\n'.format(sect_num, self.Xisection)
        txt += '{}\n'.format('=' * 68)
        txt += 'Zero-lift alpha (deg):   {:.2f}        Minimum Cd           : {:.4f}\n'.format(self.A0deg, self.CDmin)
        txt += 'd(Cl)/d(alpha)       :  {:.4f}        Cl at minimum Cd     : {:.4f}\n'.format(np.rad2deg(self.dCLdA), self.CLCDmin)
        txt += 'd(Cl)/d(alpha)@stall :  {:.3f}        d(Cd)/d(Cl**2)       : {:.4f}\n'.format(np.rad2deg(self.dCLdAstall), self.dCDdCL2)
        txt += 'Maximum Cl           :  {:.2f}         Reference Re number  :  {}\n'.format(self.CLmax, self.REref)
        txt += 'Minimum Cl           : {:.2f}         Re scaling exponent  : {:.4f}\n'.format(self.CLmin, self.REexp)
        txt += 'Cl increment to stall:  {:.3f}        Cm                   : {:.3f}\n'.format(self.CLinc2stall, self.Cmconst)
        txt += '                                     Mcrit                :  {:.3f}\n'.format(self.Mcrit)
        txt += '{}\n'.format('=' * 68)
        return txt
