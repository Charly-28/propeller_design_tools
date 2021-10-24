import propeller_design_tools as pdt


pdt.set_airfoil_database(r'D:\Python Projects\propeller_design_tools\foil_database')
pdt.set_propeller_database(r'D:\Python Projects\propeller_design_tools\prop_database')

prop = pdt.Propeller('MyPropeller')
prop.plot_geometry()
prop.generate_stl_geometry()
