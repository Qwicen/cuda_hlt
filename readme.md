CUDA HLT
========

Welcome to the CUDA High Level Trigger project, an attempt to provide
a full HLT1 realization on GPU.

How to create the input
-----------------------

In the current development stage, the input is created by running Brunel. 
On one hand, the raw bank / hit information is written to binary files; 
on the other hand, the MC truth information is written to ROOT files to be 
able to run the checker. For this, Renato Quagliani's tool PrTrackerDumper is 
used.
Use the branch 
dovombru_TrackerDumper_forGPUInput (on current master branch)
of the Rec repository to create the input by following these steps:

Login to lxplus:

    ssh -XY dovombru@lxplus7.cern.ch
    LbLogin -c x86_64-centos7-gcc7-opt

Compilation:
    lb-dev --nightly lhcb-head Brunel/HEAD
    cd BrunelDev_HEAD/
    git lb-use Rec
    git lb-checkout Rec/dovombru_TrackerDumper_forGPUInput Pr/PrPixel
    git lb-checkout Rec/dovombru_TrackerDumper_forGPUInput Pr/PrMCTools
    make
    
The files `options.py`, `upgrade-minbias-magdown.py` and `upgrade-minbias-magdown.xml`
are available in the `config` directory of the dovombru_TrackerDumper_forGPUInput
branch. Copy these to your BrunelDev_HEAD directory and use them to run Rec.
    
Running:
    
    mkdir velopix_raw
    mkdir TrackerDumper
    mkdir ut_hits
    ./run gaudirun.py options.py upgrade-minbias-magdown.py
    
One file per event is created and stored in the velopix_raw, TrackerDumper and ut_hits
directories, which need to be copied to folders in the CUDA_HLT1 project
to be used as input there. 
    

How to run it
-------------

The project requires a graphics card with CUDA support.
The build process doesn't differ from standard cmake projects:

    mkdir build
    cd build
    cmake ..
    make

There are some cmake options to configure the build process:

   * The build type can be specified to `RelWithDebInfo`, `Release` or `Debug`, e.g. `cmake -DBUILD_TYPE=Debug ..`
   * The option to run the validation can be turned on and off with `-DMC_CHECK`. 
   

The MC validation is a standalone version of the PrChecker, it was written by
Manuel Schiller, Rainer Schwemmer and Daniel Cámpora.

Some binary input files are included with the project for testing.
A run of the program with no arguments will let you know the basic options:

    Usage: ./cu_hlt
     -f {folder containing .bin files with velopix raw bank information}
     -g {folder containing .root files with velopix MC truth information}
     -e {folder containing .bin files with ut hit information}
     [-n {number of files to process}=0 (all)]
     [-t {number of threads / streams}=3]
     [-r {number of repetitions per thread / stream}=10]
     [-a {transmit host to device}=1]
     [-b {transmit device to host}=1]
     [-c {run checkers}=0]
     [-k {simplified kalman filter}=0]
     [-v {verbosity}=3 (info)]
     [-p (print rates)]

Here are some example run options:

    # Run all input files once
    ./cu_hlt -f ../minbias_raw

    # Run a total of 1000 events, round robin over the existing ones
    ./cu_hlt -f ../velopix_minbias_raw -n 1000

    # Run four streams, each with 4000 events, 20 repetitions
    ./cu_hlt -f ../velopix_minbias_raw -t 4 -n 4000 -r 20

    # Run twelve streams, each with 3500 events, 40 repetitions
    ./cu_hlt -f ../velopix_minbias_raw -n 3500 -t 12 -r 40

    # Run clustering and Velopix efficiency validations, no repetitions or multiple threads needed
    # Note: cu_hlt must have been compiled with -DMC_CHECK
    ./cu_hlt -f ../velopix_minbias_raw -g ../TrackerDumper -e ../ut_hits -n 10 -t 1 -r 1 -c 1
