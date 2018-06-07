/**
 *      CUDA HLT1
 *      
 *      author  -  GPU working group
 *      e-mail  -  lhcb-parallelization@cern.ch
 *
 *      Original development June, 2014
 *      Restarted development on February, 2018
 *      CERN
 */

#include <iostream>
#include <string>
#include <cstring>
#include <exception>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <unistd.h>
#include "tbb/tbb.h"
#include "cuda_runtime.h"
#include "../include/CudaCommon.h"
#include "../include/Logger.h"
#include "../include/Tools.h"
#include "../include/Timer.h"
#include "../../stream/sequence/include/Stream.cuh"
#include "../../stream/sequence/include/InitializeConstants.cuh"
#include "../../x86/velo/clustering/include/Clustering.h"

void printUsage(char* argv[]){
  std::cerr << "Usage: "
    << argv[0]
    << std::endl << " -f {folder containing .bin files with raw bank information}"
    << std::endl << (do_mc_check ? " " : " [") << "-g {folder containing .bin files with MC truth information}"
    << (do_mc_check ? "" : " ]")
    << std::endl << " [-n {number of files to process}=0 (all)]"
    << std::endl << " [-t {number of threads / streams}=3]"
    << std::endl << " [-r {number of repetitions per thread / stream}=10]"
    << std::endl << " [-a {transmit host to device}=1]"
    << std::endl << " [-b {transmit device to host}=1]"
    << std::endl << " [-c {run checkers}=0]"
    << std::endl << " [-k {simplified kalman filter}=0]"
    << std::endl << " [-v {verbosity}=3 (info)]"
    << std::endl << " [-p (print rates)]"
    << std::endl;
}

int main(int argc, char *argv[])
{
  std::string folder_name_raw;
  std::string folder_name_MC = "";
  uint number_of_files = 0;
  uint tbb_threads = 3;
  uint number_of_repetitions = 10;
  uint verbosity = 3;
  bool print_individual_rates = false;
  bool transmit_host_to_device = true;
  bool transmit_device_to_host = true;
  bool do_check = false;
  bool do_simplified_kalman_filter = false;
   
  signed char c;
  while ((c = getopt(argc, argv, "f:g:n:t:r:pha:b:d:v:c:k:")) != -1) {
    switch (c) {
    case 'f':
      folder_name_raw = std::string(optarg);
      break;
    case 'g':
      folder_name_MC = std::string(optarg);
      break;
    case 'n':
      number_of_files = atoi(optarg);
      break;
    case 't':
      tbb_threads = atoi(optarg);
      break;
    case 'r':
      number_of_repetitions = atoi(optarg);
      break;
    case 'a':
      transmit_host_to_device = atoi(optarg);
      break;
    case 'b':
      transmit_device_to_host = atoi(optarg);
      break;
    case 'c':
      do_check = atoi(optarg);
      break;
    case 'k':
      do_simplified_kalman_filter = atoi(optarg);
      break;
    case 'v':
      verbosity = atoi(optarg);
      break;
    case 'p':
      print_individual_rates = true;
      break;
    case '?':
    case 'h':
    default:
      printUsage(argv);
      return -1;
    }
  }

  // Check how many files were specified and
  // call the entrypoint with the suggested format
  if(folder_name_raw.empty()){
    std::cerr << "No folder specified" << std::endl;
    printUsage(argv);
    return -1;
  }

  if(folder_name_MC.empty() && do_mc_check){
    std::cerr << "No MC folder specified, but MC CHECK turned on" << std::endl;
    printUsage(argv);
    return -1;
  }

  // Set verbosity level
  std::cout << std::fixed << std::setprecision(6);
  logger::ll.verbosityLevel = verbosity;

  // Get device properties
  cudaDeviceProp device_properties;
  cudaCheck(cudaGetDeviceProperties(&device_properties, 0));

  // Show call options
  std::cout << "Requested options:" << std::endl
    << " folder with raw bank input (-f): " << folder_name_raw << std::endl
    << " folder with MC truth input (-g): " << folder_name_MC << std::endl
    << " MC check (compile opt): " << do_mc_check << std::endl
    << " number of files (-n): " << number_of_files << std::endl
    << " tbb threads (-t): " << tbb_threads << std::endl
    << " number of repetitions (-r): " << number_of_repetitions << std::endl
    << " transmit host to device (-a): " << transmit_host_to_device << std::endl
    << " transmit device to host (-b): " << transmit_device_to_host << std::endl
    << " run checkers (-c): " << do_check << std::endl
    << " simplified kalman filter (-k): " << do_simplified_kalman_filter << std::endl
    << " print rates (-p): " << print_individual_rates << std::endl
    << " verbosity (-v): " << verbosity << std::endl
    << " device: " << device_properties.name << std::endl
    << std::endl;

  // Read folder contents
  std::vector<char> velopix_events;
  std::vector<uint> velopix_event_offsets;
  std::vector<char> ut_events;
  std::vector<uint> ut_event_offsets;
  readFolder(
    folder_name_raw,
    number_of_files,
	  velopix_events,
    velopix_event_offsets
  );

  std::vector<char> geometry;
  readGeometry(folder_name_raw, geometry);

  // Copy data to pinned host memory
  char* host_velopix_events_pinned;
  uint* host_velopix_event_offsets_pinned;
  cudaCheck(cudaMallocHost((void**)&host_velopix_events_pinned, velopix_events.size()));
  cudaCheck(cudaMallocHost((void**)&host_velopix_event_offsets_pinned, velopix_event_offsets.size() * sizeof(uint)));
  std::copy_n(std::begin(velopix_events), velopix_events.size(), host_velopix_events_pinned);
  std::copy_n(std::begin(velopix_event_offsets), velopix_event_offsets.size(), host_velopix_event_offsets_pinned);

  // Initialize detector constants on GPU
  initializeConstants();

  // Create streams
  const auto number_of_events = velopix_event_offsets.size() - 1;
  std::vector<Stream> streams (tbb_threads);
  for (int i=0; i<streams.size(); ++i) {
    streams[i].initialize(
      velopix_events,
      velopix_event_offsets,
      geometry,
      number_of_events,
      transmit_host_to_device,
      transmit_device_to_host,
      do_check,
      do_simplified_kalman_filter,
      print_individual_rates,
      folder_name_MC,
      i
    );

    // Memory consumption
    size_t free_byte ;
    size_t total_byte ;
    cudaCheck( cudaMemGetInfo( &free_byte, &total_byte ) );
    float free_percent = (float)free_byte / total_byte * 100;
    float used_percent = (float)(total_byte - free_byte) / total_byte * 100;
    verbose_cout << "GPU memory: " << free_percent << " percent free, "
      << used_percent << " percent used " << std::endl;
  }
  
  // Attempt to execute all in one go
  Timer t;
  tbb::parallel_for(
    static_cast<uint>(0),
    static_cast<uint>(tbb_threads),
    [&] (uint i) {
      auto& s = streams[i];
      s(
        host_velopix_events_pinned,
        host_velopix_event_offsets_pinned,
        velopix_events.size(),
        velopix_event_offsets.size(),
        number_of_events,
        number_of_repetitions
      );
    }
  );
  t.stop();

  std::cout << (number_of_events * tbb_threads * number_of_repetitions / t.get()) << " events/s" << std::endl
    << "Ran test for " << t.get() << " seconds" << std::endl;

  // Free and reset device
  // cudaCheck(cudaFreeHost(host_velopix_events_pinned));
  // cudaCheck(cudaFreeHost(host_velopix_event_offsets_pinned));
  cudaCheck(cudaDeviceReset());

  return 0;
}
