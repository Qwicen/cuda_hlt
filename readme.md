CUDA HLT
========

Welcome to the CUDA High Level Trigger project, an attempt to provide
a full HLT1 realization on GPU.

How to run it
-------------

The project requires a graphics card with CUDA support.
The build process doesn't differ from standard cmake projects:

    mkdir build
    cd build
    cmake ..
    make

There are some cmake options to configure the build process:

   * The build type can be specified to `RelWithDebInfo`, `Release` or `Debug`, e.g. `cmake -DCMAKE_BUILD_TYPE=Debug ..`
   * The option to run the validation, on by default, can be turned off with `-DMC_CHECK=Off`. 
   
Some binary input files are included with the project for testing.
A run of the program with no arguments will let you know the basic options:

    Usage: ./cu_hlt
     -f {folder containing .bin files with raw bank information}
     -g {folder containing .bin files with MC truth information}
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

    # Run all input files once with the tracking validation
    ./cu_hlt -f ../velopix_minbias_raw -g ../velopix_minbias_MC

    # Note: For the examples below, cu_hlt must have been compiled with -DMC_CHECK=Off
    # Run a total of 1000 events, round robin over the existing ones
    ./cu_hlt -f ../velopix_minbias_raw -n 1000

    # Run four streams, each with 4000 events, 20 repetitions
    ./cu_hlt -f ../velopix_minbias_raw -t 4 -n 4000 -r 20

    # Run twelve streams, each with 3500 events, 40 repetitions
    ./cu_hlt -f ../velopix_minbias_raw -n 3500 -t 12 -r 40

    # Run one stream and print all memory allocations
    ./cu_hlt -f ../velopix_minbias_raw -n 5000 -t 1 -r 1 -p
