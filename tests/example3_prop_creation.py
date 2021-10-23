import propeller_design_tools as pdt


pdt.set_airfoil_database(r'D:\Python Projects\propeller_design_tools\foil_database')
pdt.set_propeller_database(r'D:\Python Projects\propeller_design_tools\prop_database')

prop = pdt.create_propeller(
    name='MyPropeller',
    nblades=3,
    radius=0.5,
    hub_radius=0.06,
    hub_wake_disp_br=0.06,
    design_speed=15,
    design_cl={'const': 0.18},
    design_atmo_props={'altitude': 5},
    design_vorform='vrtx',
    design_rpm=1800,
    design_power=400,
    n_radial=50,
    station_params={0.75: 'clarky'},
    geo_params={'tot_skew': 15, 'n_prof_pts': None, 'n_profs': 50},
)
