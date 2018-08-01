to do:

* load exact UT layer positions (hard-coded at the moment)
* implement extrapolators and deflections tables
* check results with nominal version


VectorClass was taken from https://gitlab.cern.ch/lhcb/LHCb/tree/master/Kernel/VectorClass/VectorClass


In the VeloUT algorithm two look-up tables (LUTs) are used to get the deflection for the search windows and the integrated Bdl to estimate the momentum and are saved in the files bdl.txt and deflection.txt.

The Bdl LUT can be obtained by compiling the test.cpp program, by which it is dumped.

Both LUTs can be obtained from the nominal LHCb software by e.g. editing Pr/PrVeloUT/src/PrVeloUT.cpp : The LUTs are used there as vectors 'bdl' and 'fudgeFactors', where one can add a line of code to loop over them and dump their values. As an example, see /afs/cern.ch/work/f/freiss/public/recept/VP-Studies/MyBrunel/Pr/PrVeloUT/src/PrVeloUT.cpp line 110.



