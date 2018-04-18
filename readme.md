Search by triplet
=================

Welcome to the search by triplet algorithm written in CUDA.

Here is some documentation for the algorithm idea implemented here:

* https://cernbox.cern.ch/index.php/s/R8i13RP6uLm9JJW

How to run it
-------------

The project requires a graphics card with CUDA support. The build process doesn't differ from standard cmake projects:

    mkdir build
    cmake ..
    make

Some binary input files are included with the project for testing. A run of the program with no arguments will let you know the basic options:

    Usage: ./cu_hlt
     -f {folder containing .bin files}
     [-n {number of files to process}=0 (all)]
     [-t {number of threads / streams}=3]
     [-r {number of repetitions per thread / stream}=10]
     [-a {transmit host to device}=1]
     [-b {transmit device to host}=1]
     [-c {consolidate tracks}=0]
     [-k {simplified kalman filter}=0]
     [-v {verbosity}=3 (info)]
     [-p (print rates)]


Here are some example run options:

    # Run all input files once
    ./cu_hlt -f ../input

    # Run a total of 1000 events, round robin over the existing ones
    ./cu_hlt -f ../input -n 1000

    # Run four streams, each with 4000 events, 20 repetitions
    ./cu_hlt -f ../input -t 4 -n 4000 -r 20

    # Run twelve streams, each with 3500 events, 40 repetitions
    ./cu_hlt -f ../input -n 3500 -t 12 -r 40
