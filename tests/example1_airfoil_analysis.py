import propeller_design_tools as pdt


pdt.set_airfoil_database(r'D:\Python Projects\propeller_design_tools\foil_database')
# pdt.clear_foil_database()

foil = pdt.Airfoil(name='clarky')
foil.plot_geometry()

# foil.calculate_xfoil_polars(re=[5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 1e8])
fig, ax = foil.plot_polar_data('CD', 'CL')

pol = foil.interpolate_polar(npts=200, re=5e7, mach=0, ncrit=9)
ax.plot(pol['CD'], pol['CL'], marker='o')   # why does this come out a little bit "wavy"?
