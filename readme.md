CUDA HLT
========

Welcome to the CUDA High Level Trigger project, an attempt to provide
a full HLT1 realization on GPU.

Requisites
----------
The project requires a graphics card with CUDA support, CUDA 9.2 and a compiler supporting C++14. It also requires the developer package of `tbb`.

If you are working from a node with CVMFS and CentOS 7, we suggest the following setup:

```shell
export CPATH=/cvmfs/lhcb.cern.ch/lib/lcg/releases/tbb/44_20160413-f254c/x86_64-centos7-gcc7-opt/include:$CPATH
export LD_LIBRARY_PATH=/cvmfs/lhcb.cern.ch/lib/lcg/releases/tbb/44_20160413-f254c/x86_64-centos7-gcc7-opt/lib:$LD_LIBRARY_PATH
source /cvmfs/lhcb.cern.ch/lib/lcg/releases/gcc/7.3.0/x86_64-centos7/setup.sh
```

Regardless of the OS you are running on, you can check your compiler versions as follows:

```shell
$ g++ --version
g++ (GCC) 7.3.0

$ nvcc --version
Cuda compilation tools, release 9.2, V9.2.88
```

You can check your compiler standard compatibility by scrolling to the `C++14 features` chart [here](https://en.cppreference.com/w/cpp/compiler_support).

Optional: you can compile the project with ROOT. Then, trees will be filled with variables to check when running the VeloUT algorithm on x86 architecture.

Where to find input
-------------
Input from 10k events can be found here: /afs/cern.ch/work/d/dovombru/public/gpu_input/10kevents

How to run it
-------------

The build process doesn't differ from standard cmake projects:

    mkdir build
    cd build
    cmake ..
    make

There are some cmake options to configure the build process:

   * The build type can be specified to `RelWithDebInfo`, `Release` or `Debug`, e.g. `cmake -DCMAKE_BUILD_TYPE=Debug ..`
   * The option to run the validation, on by default, can be turned off with `-DMC_CHECK=Off`. 
   

The MC validation is a standalone version of the PrChecker, it was written by
Manuel Schiller, Rainer Schwemmer and Daniel Cámpora.

Some binary input files are included with the project for testing.
A run of the program with no arguments will let you know the basic options:

    Usage: ./cu_hlt
    -f {folder containing .bin files with raw bank information}
    -d {folder containing .bin files with MC truth information}
    -e {folder containing bin files with UT hit information}
    -g {folder containing geometry descriptions}
    -n {number of events to process}=0 (all)
    -o {offset of events from which to start}=0 (beginning)
    -t {number of threads / streams}=1
    -r {number of repetitions per thread / stream}=1
    -b {transmit device to host}=1
    -c {run checkers}=0
    -k {simplified kalman filter}=0
    -m {reserve Megabytes}=1024
    -v {verbosity}=3 (info)
    -p (print memory usage)
    -x {run algorithms on x86 architecture as well (if possible)}=0


Here are some example run options:

    # Run all input files once with the tracking validation
    ./cu_hlt -f ../input/minbias/velopix_raw -e ../input/minbias/ut_hits -d ../input/minbias/MC_info -g ../input/geometry

    # Note: For the examples below, cu_hlt must have been compiled with -DMC_CHECK=Off
    # Run a total of 1000 events, round robin over the existing ones
    ./cu_hlt -f ../input/minbias/velopix_raw -e ../input/minbias/ut_hits -g ../input/geometry -n 1000

    # Run four streams, each with 4000 events, 20 repetitions
    ./cu_hlt -f ../input/minbias/velopix_raw -e ../input/minbias/ut_hits -g ../input/geometry -t 4 -n 4000 -r 20

    # Run twelve streams, each with 3500 events, 40 repetitions
    ./cu_hlt -f ../input/minbias/velopix_raw -e ../input/minbias/ut_hits -g ../input/geometry -n 3500 -t 12 -r 40

    # Run clustering and Velopix efficiency validations, no repetitions or multiple threads needed
    # Note: cu_hlt must have been compiled with -DMC_CHECK
    ./cu_hlt -f ../input/minbias/velopix_raw -d ../input/minbias/MC_info -e ../input/minbias/ut_hits -g ../input/geometry/ -n 10 -t 1 -r 1 -c 1
    
    # Run one stream and print all memory allocations
    ./cu_hlt -f ../input/minbias/velopix_raw -e ../input/minbias/ut_hits -g ../input/geometry -n 5000 -t 1 -r 1 -p

[This readme](readme_cuda_developer.md) explains how to add a new algorithm to the sequence and how to use the memory scheduler to define globab memory variables for this sequence and pass on the dependencies.