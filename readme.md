Allen
=====

Welcome to Allen, a project with the aim to provide a full HLT1 realization on GPU.

Requisites
----------
The project requires a graphics card with CUDA support, CUDA 10.0, CMake 3.12 and a compiler supporting C++14.

If you are working from a node with CVMFS and CentOS 7, we suggest the following setup:

```shell
source /cvmfs/lhcb.cern.ch/lib/lcg/releases/gcc/7.3.0/x86_64-centos7/setup.sh
export PATH=/cvmfs/lhcb.cern.ch/lib/contrib/CMake/3.12.1/Linux-x86_64/bin/:$PATH
export PATH=/usr/local/cuda/bin:$PATH
```

Regardless of the OS you are running on, you can check your compiler versions as follows:

```shell
$ g++ --version
g++ (GCC) 7.3.0

$ nvcc --version
Cuda compilation tools, release 10.0, V10.0.130

$ cmake --version
cmake version 3.12.1
```

You can check your compiler standard compatibility by scrolling to the `C++14 features` chart [here](https://en.cppreference.com/w/cpp/compiler_support).

Optionally you can compile the project with ROOT. Then, trees will be filled with variables to check when running the UT tracking or SciFi tracking algorithms on x86 architecture.
In addition, histograms of reconstructible and reconstructed tracks are then filled in the track checker. For more details on how to use them to produce plots of efficiencies, momentum resolution etc. see [this readme](checker/tracking/readme.md). 

You can setup ROOT in CVMFS as follows:

```shell
source /cvmfs/lhcb.cern.ch/lib/lcg/releases/ROOT/6.08.06-d7e12/x86_64-centos7-gcc62-opt/bin/thisroot.sh
```

[Building and running inside Docker](readme_docker.md)

Where to find input
-------------
Input from 1k events can be found here:

* minimum bias (for performance checks): `/afs/cern.ch/work/d/dovombru/public/gpu_input/1kevents_minbias.tar.gz`
* Bs->PhiPhi (for efficiency checks): `/afs/cern.ch/work/d/dovombru/public/gpu_input/1kevents_BsPhiPhi.tar.gz`
* J/Psi->MuMu (for muon efficiency checks): `/afs/cern.ch/work/d/dovombru/public/gpu_input/10kevents_JPsiMuMu_private.tar.gz`

How to build it
---------------

The build process doesn't differ from standard cmake projects:

    mkdir build
    cd build
    cmake ..
    make

There are some cmake options to configure the build process:

* The sequence can be configured by specifying `-DSEQUENCE=<name_of_sequence>`. For a complete list of sequences available, check `configuration/sequences/`. Sequence names should be specified without the `.h`, ie. `-DSEQUENCE=VeloPVUTSciFiDecoding`.
* The build type can be specified to `RelWithDebInfo`, `Release` or `Debug`, e.g. `cmake -DCMAKE_BUILD_TYPE=Debug ..`
* If ROOT is available, it can be enabled to generate graphs by `-DUSE_ROOT=ON`
* If more verbose build output from the CUDA toolchain is desired, specify `-DCUDA_VERBOSE_BUILD=ON`
* If multiple versions of CUDA are installed and CUDA 10.0 is not the default, it can be specified using: `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.0/bin/nvcc`
* The MC validation is standalone, it was written by Manuel Schiller, Rainer Schwemmer, Daniel Cámpora and Dorothea vom Bruch.

How to run it
-------------

Some binary input files are included with the project for testing.
A run of the program with no arguments will let you know the basic options:

    Usage: ./Allen
    -f {folder containing directories with raw bank binaries for every sub-detector}
    -b {folder containing .bin files with muon common hits}
    --mdf {use MDF files as input instead of binary files}
    -g {folder containing detector configuration}
    -d {folder containing .bin files with MC truth information}
    -n {number of events to process}=0 (all)
    -o {offset of events from which to start}=0 (beginning)
    -t {number of threads / streams}=1
    -r {number of repetitions per thread / stream}=1
    -c {run checkers}=0
    -m {reserve Megabytes}=1024
    -v {verbosity}=3 (info)
    -p {print memory usage}=0
    -a {run only data preparation algorithms: decoding, clustering, sorting}=0

Here are some example run options:

    # Run all input files once with the tracking validation
    ./Allen

    # Specify input files, run once over all of them with tracking validation
    ./Allen -f ../input/minbias/banks/ -d ../input/minbias/MC_info/ -b ../input/minbias/muon_common_hits

    # Run a total of 1000 events, round robin over the existing ones, without tracking validation
    ./Allen -c 0 -n 1000

    # Run four streams, each with 4000 events, 20 repetitions
    ./Allen -t 4 -n 4000 -r 20 -c 0

    # Run one stream and print all memory allocations
    ./Allen -n 5000 -p


[This readme](contributing.md) explains how to add a new algorithm to the sequence and how to use the memory scheduler to define global memory variables for this sequence and pass on the dependencies. It also explains which checks to do before placing a merge request with your changes.
