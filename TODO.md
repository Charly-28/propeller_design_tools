PDT To-Do List
==============
High-Priority (soon)
--------------------
1. Make PDT create the user-set directories on setting if they don't exist yet
2. Make the main Propeller() plot number into Newtons/kW

Backlog (med-priority)
----------------------
* Is atmo_props['altitude'] input in km or m?
* Determine best way to indicate all input paramter units & implement
* Add a "Cm_const" callout to the RadialStation plot
* Work on better / non-buggy implementation of > 1 RadialStations in propeller creation
* Consider a 1x iterative scheme to "go back" and re-calculate stations parameters
after a successful Propeller() creation returned form XROTOR, and we actually know what
the chords of each station are, so we can update the "1/10th of the local radius" estimate
* Work on propeller "analysis sweeps" to gain access to more of the XROTOR functionality
(example5_prop_analysis.py)
* Work on propeller optimization routines (example6_grid_optimizer.py)

Wishlist (low-priority)
-----------------------
* Get Gud at coding - obtain & integrate XFOIL and XROTOR source code such that 
the user no longer needs to get the executables themselves, the entire package is
finally self-contained
* Consider ways to automate the downloading of coordinate files
* Consider ways to automate the generation of airfoil polar data
* Consider ways to generate hub and blade/hub interface geometries
* Consider ways to "cut in" in the blade chords near the hub when using 
design_vorform='pot'