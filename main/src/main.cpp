/**
 *      CUDA Search by Triplet project
 *      
 *      author  -   Daniel Campora
 *      email   -   dcampora@cern.ch
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
#include "../include/Common.h"
#include "../include/Logger.h"
#include "Tools.h"
#include "Stream.cuh"
#include "Timer.h"

void printUsage(char* argv[]){
  std::cerr << "Usage: "
    << argv[0]
    << std::endl << " -f {folder containing .bin files}"
    << std::endl << " [-n {number of files to process}=0 (all)]"
    << std::endl << " [-t {number of threads / streams}=3]"
    << std::endl << " [-r {number of repetitions per thread / stream}=10]"
    << std::endl << " [-a {transmit host to device}=1 (-a 0 implies -r 1)]"
    << std::endl << " [-b {transmit device to host}=0]"
    << std::endl << " [-p (print individual rates)]"
    << std::endl;
}

int main(int argc, char *argv[])
{
  std::string folder_name;
  unsigned int number_of_files = 0;
  unsigned int tbb_threads = 3;
  unsigned int number_of_repetitions = 10;
  bool print_individual_rates = false;
  bool transmit_host_to_device = true;
  bool transmit_device_to_host = false;

  signed char c;
  while ((c = getopt(argc, argv, "f:n:t:r:pha:b:d:")) != -1) {
    switch (c) {
    case 'f':
      folder_name = std::string(optarg);
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
    case 'p':
      print_individual_rates = true;
      break;
    case 'a':
      transmit_host_to_device = atoi(optarg);
      break;
    case 'b':
      transmit_device_to_host = atoi(optarg);
      break;
    case '?':
    case 'h':
    default:
      printUsage(argv);
      return -1;
    }
  }

  // Set device
  cudaDeviceProp device_properties;
  cudaCheck(cudaGetDeviceProperties(&device_properties, 0));

  // If there is no transmission from host to device,
  // the data will be invalidated after the first iteration,
  if (transmit_host_to_device == false) {
    // Restrict number of iterations to 1
    number_of_repetitions = 1;
  }

  // Check how many files were specified and
  // call the entrypoint with the suggested format
  if(folder_name.empty()){
    std::cerr << "No folder specified" << std::endl;
    printUsage(argv);
    return -1;
  }
  
  // Show call options
  std::cout << "Requested options:" << std::endl
    << " folder (-f): " << folder_name << std::endl
    << " number of files (-n): " << number_of_files << std::endl
    << " tbb threads (-t): " << tbb_threads << std::endl
    << " number of repetitions (-r): " << number_of_repetitions << std::endl
    << " transmit host to device (-a): " << transmit_host_to_device << std::endl
    << " transmit device to host (-b): " << transmit_device_to_host << std::endl
    << " print rates (-p): " << print_individual_rates << std::endl
    << " device: " << device_properties.name << std::endl
    << std::endl;

  // Read folder contents
  std::vector<char> events;
  std::vector<unsigned int> event_offsets;
  std::vector<unsigned int> hit_offsets;
  readFolder(folder_name, number_of_files, events, event_offsets, hit_offsets);

  // Set verbosity to max
  std::cout << std::fixed << std::setprecision(6);
  logger::ll.verbosityLevel = 3;

  // Show some statistics
  statistics(events, event_offsets);

  // Copy data to pinned host memory
  char* host_events_pinned;
  unsigned int* host_event_offsets_pinned;
  unsigned int* host_hit_offsets_pinned;
  cudaCheck(cudaMallocHost((void**)&host_events_pinned, events.size()));
  cudaCheck(cudaMallocHost((void**)&host_event_offsets_pinned, event_offsets.size() * sizeof(unsigned int)));
  cudaCheck(cudaMallocHost((void**)&host_hit_offsets_pinned, hit_offsets.size() * sizeof(unsigned int)));
  std::copy_n(std::begin(events), events.size(), host_events_pinned);
  std::copy_n(std::begin(event_offsets), event_offsets.size(), host_event_offsets_pinned);
  std::copy_n(std::begin(hit_offsets), hit_offsets.size(), host_hit_offsets_pinned);

  // Create streams
  const auto number_of_events = event_offsets.size();
  std::vector<Stream> streams (tbb_threads);
  for (int i=0; i<streams.size(); ++i) {
    streams[i].initialize(
      events,
      event_offsets,
      hit_offsets,
      number_of_events,
      events.size(),
      transmit_host_to_device,
      transmit_device_to_host,
      i,
      print_individual_rates
    );
  }

  // Attempt to execute all in one go
  Timer t;
  tbb::parallel_for(
    static_cast<unsigned int>(0),
    static_cast<unsigned int>(tbb_threads),
    [&] (unsigned int i) {
      auto& s = streams[i];
      s(
        host_events_pinned,
        host_event_offsets_pinned,
        host_hit_offsets_pinned,
        events.size(),
        event_offsets.size(),
        hit_offsets.size(),
        0,
        event_offsets.size(),
        number_of_repetitions
      );
    }
  );
  t.stop();

  std::cout << (event_offsets.size() * tbb_threads * number_of_repetitions / t.get()) << " events/s" << std::endl
    << "Ran test for " << t.get() << " seconds" << std::endl;

  // Free and reset device
  cudaCheck(cudaFreeHost(host_events_pinned));
  cudaCheck(cudaFreeHost(host_event_offsets_pinned));
  cudaCheck(cudaFreeHost(host_hit_offsets_pinned));
  cudaCheck(cudaDeviceReset());

  return 0;
}
