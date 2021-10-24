propeller_design_tools (PDT)
============================
---
**Work in progress / incomplete documentation**

---

Description
===========
Python 3.7 package that provides exactly what it sounds 
like by automating usage of the GPL-licensed 
CLI-utilities XFOIL and XROTOR.

Both utilities are published by professor Mark Drela (MIT).
- XFOIL: for arbitrary 2D airfoil analysis
- XROTOR: for arbitrary propeller design schemes

Purpose
=======
PDT seeks to provide the user a set of python3 utilities
that can be used for arbitrary scripting efforts to automate
usage of both XFOIL and XROTOR while implementing its own 
unique python3.7-native algorithms to maintain local
input files, meta files, databases, and results files and
weave everything together for the user in a simple,
meaningful way to aid in the initial / investigatory 
stage of well-behaved propeller designs.

Getting Started
===============
Installation
------------
`pip install propeller_design_tools`

General Operation
-----------------
`import propeller_design_tools as pdt`

PDT operates on two different "database" directories, defined
by the user with:

    pdt.set_airfoil_database(path: str)
    pdt.set_propeller_database(path: str)

**The user must set these two directories at the top 
of every script right after the imports**

*The airfoil directory will be used to store any foil / 
XFOIL- related support files, and the propeller directory
will be used similarly to store any propeller / XROTOR - 
related support files.*

Pre-Requisite: XFOIL and XROTOR Executables
-------------------------------------------
In order to utilize any PDT functionality that depends on 
running XFOIL, the "xfoil.exe" executable file needs to be
in the user-set "airfoil_database" location. 

[XFOIL executable and docs](https://web.mit.edu/drela/Public/web/xfoil/)

Likewise, in order to utilize any PDT functionality that
depends on running XROTOR, the "xrotor.exe" executable file
needs to be in the user-set "propeller_database" location.

[XROTOR executable and docs](http://www.esotec.org/sw/crotor.html#download)
*(this is actually a link to "CROTOR", which I find is
actually the easiest way to obtain a windows-executable
of XROTOR)*

Example Scripts / Workflow
--------------------------
At a high-level, the current concept for PDT workflow is as 
follows (after obtaining the required executables and pip-installing 
the PDT package):

1. Obtain normalized airfoil coordinate files from
[UIUC Database](https://m-selig.ae.illinois.edu/ads/coord_database.html)
-> save these files into the "airfoil_database" directory


2. Use PDT to run XFOIL across ranges of Reynolds Numbers in order to
populate database data for the desired foil sections -> see 
[example1_airfoil_analysis.py](
   https://github.com/helloDestroyerOfWorlds/propeller_design_tools/blob/master/tests/example1_airfoil_analysis.py
   )

   ![ex1-1.png](https://raw.githubusercontent.com/helloDestroyerOfWorlds/propeller_design_tools/master/tests/ex1-1.png)
   ![ex1-2.png](https://raw.githubusercontent.com/helloDestroyerOfWorlds/propeller_design_tools/master/tests/ex1-2.png)


3. Once the required 2D airfoil data is generated, PDT can then be used
to automatically generate all the required 2D foil definition parameters
required by XROTOR (these "station parameters" are essentially what 
allow XROTOR to model the performance of well-behaved, arbitrarily-lofted 
blade geometries) -> see
[example2_radialstation_creation.py](
   https://github.com/helloDestroyerOfWorlds/propeller_design_tools/blob/master/tests/example2_radialstation_creation.py
   )

   ![ex2-1.png](https://raw.githubusercontent.com/helloDestroyerOfWorlds/propeller_design_tools/master/tests/ex2-1.png)
   
   But this step is also automated & displayed by PDT when the user uses
the builtin PDT propeller creation function -> see
[example3_prop_creation.py](
   https://github.com/helloDestroyerOfWorlds/propeller_design_tools/blob/master/tests/example3_prop_creation.py
   )

   ![ex3-1.png](https://raw.githubusercontent.com/helloDestroyerOfWorlds/propeller_design_tools/master/tests/ex3-1.png)
   ![ex3-2.png](https://raw.githubusercontent.com/helloDestroyerOfWorlds/propeller_design_tools/master/tests/ex3-2.png)


4. PDT's Propeller() object instances can generate 3D geometry files 
including profle xyz coordinate listings, and .stl 3D geometry files -> see
[example4_stl_generation.py](
   https://github.com/helloDestroyerOfWorlds/propeller_design_tools/blob/master/tests/example4_stl_generation.py
   )

   ![ex4-1.png](https://raw.githubusercontent.com/helloDestroyerOfWorlds/propeller_design_tools/master/tests/ex4-1.png)


5. **WIP** Prop analysis (integrating XROTOR's sweeps commands and displaying 
outputs)


6. **WIP** Prop optimization (grid-search style generic optimizer for "optimal"
prop design generation by means of maximizing or minimizing a given output / 
calculated metric based on outputs, optionally taking into account different
propeller operating points via the ability to define the propeller's "duty-cycle")