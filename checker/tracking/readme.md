Track Checker
==============

For producing plots of efficiencies etc., Allen needs to be compiled with ROOT (see main [readme](../../readme.md) ). 
Create the directory Allen/output, then the ROOT file PrCheckerPlots.root will be saved there when running the MC validation (option -c).

Efficiency plots
------------------------
Histograms of reconstructible and reconstructed tracks are saved in `Allen/output/PrCheckerPlots.root`.
Plots of efficiencies versus various kinematic variables can be created by running `efficiency_plots.py` in the directory 
`checker/tracking/python_scripts`. The resulting ROOT file `efficiency_plots.root` with graphs of efficiencies is saved in the directory `python_scripts`.


Momentum resolution plots
--------------------------
A 2D histogram of momentum resolution versus momentum is also stored in `Allen/output/PrCheckerPlots.root` for Upstream and Forward tracks. 
Velo tracks are straight lines, so no momentum resolution is calculated. Running the script `momentum_resolution.py` in the directory `checker/tracking/python_scripts` 
will produce a plot of momentum resolution versus momentum in the ROOT file `momentum_resolution.root` in the directory `python_scripts`. 
In this script, the 2D histogram of momentum resolution versus momentum is projected onto the momentum resolution axis in slices of the momentum. 
The resulting 1D histograms are fitted with a Gaussian function if they have more than 100 entries. The Gaussian fit is constrained to the region [-0.05,0.05] in 
the case of Forward tracks and to [-0.5, 0.5] for Upstream tracks respectively to avoid the non-Gaussian tails. (It is to be studied whether we need a different fit function). 
The mean and sigma of the Gaussian are used as value and uncertainty in the momentum resolution versus momentum plot. 
The plot is only generated if at least one momentum slice histogram has more than 100 entries.