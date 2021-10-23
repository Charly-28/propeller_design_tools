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

`pdt.set_airfoil_database(path: str)`
`pdt.set_propeller_database(path: str)`

**The user must set these two directories at the top 
of every script right after the imports**

*The airfoil directory will be used to store any foil / 
XFOIL- related support files, and the propeller directory
will be used similarly to store any propeller / XROTOR - 
related support files.*

XFOIL and XROTOR executables
----------------------------
In order to utilize any PDT functionality that depends on 
running XFOIL, the "xfoil.exe" executable file needs to be
in the user-set "airfoil_database" location. 

[XFOIL executable and docs](https://web.mit.edu/drela/Public/web/xfoil/)

Likewise, in order to utilize any PDT functionality that
depends on running XROTOR, the "xrotor.exe" executable file
needs to be in the user-set "propeller_database" location.

[XROTOR executable and docs](http://www.esotec.org/sw/crotor.html#download)
