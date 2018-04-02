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
#include "../include/Tools.h"
#include "../../stream/include/Stream.cuh"
#include "../include/Timer.h"
#include "../../x86/include/Clustering.h"

void printUsage(char* argv[]){
  std::cerr << "Usage: "
    << argv[0]
    << std::endl << " -f {folder containing .bin files}"
    << std::endl << " [-n {number of files to process}=0 (all)]"
    << std::endl << " [-t {number of threads / streams}=3]"
    << std::endl << " [-r {number of repetitions per thread / stream}=10]"
    << std::endl << " [-a {transmit host to device}=1 (-a 0 implies -r 1)]"
    << std::endl << " [-b {transmit device to host}=1]"
    << std::endl;
}

int main(int argc, char *argv[])
{
  std::string folder_name;
  unsigned int number_of_files = 0;
  unsigned int tbb_threads = 3;
  unsigned int number_of_repetitions = 10;
  unsigned int verbosity = 1;
  bool print_individual_rates = false;
  bool transmit_host_to_device = true;
  bool transmit_device_to_host = true;

  signed char c;
  while ((c = getopt(argc, argv, "f:n:t:r:pha:b:d:v:")) != -1) {
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
    case 'a':
      transmit_host_to_device = atoi(optarg);
      break;
    case 'b':
      transmit_device_to_host = atoi(optarg);
      break;
    case 'v':
      verbosity = atoi(optarg);
      break;
    case '?':
    case 'h':
    default:
      printUsage(argv);
      return -1;
    }
  }

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

  // Set verbosity level
  logger::ll.verbosityLevel = verbosity;

  // Get device properties
  cudaDeviceProp device_properties;
  cudaCheck(cudaGetDeviceProperties(&device_properties, 0));

  // Show call options
  std::cout << "Requested options:" << std::endl
    << " folder (-f): " << folder_name << std::endl
    << " number of files (-n): " << number_of_files << std::endl
    << " tbb threads (-t): " << tbb_threads << std::endl
    << " number of repetitions (-r): " << number_of_repetitions << std::endl
    << " transmit host to device (-a): " << transmit_host_to_device << std::endl
    << " transmit device to host (-b): " << transmit_device_to_host << std::endl
    << " print rates (-p): " << print_individual_rates << std::endl
    << " verbosity (-v): " << verbosity << std::endl
    << " device: " << device_properties.name << std::endl
    << std::endl;

  // Read folder contents
  std::vector<char> events;
  std::vector<unsigned int> event_offsets;
  readFolder(folder_name, number_of_files, events, event_offsets);

  std::vector<char> geometry;
  readGeometry(folder_name, geometry);

  // Invoke clustering
  std::vector<std::vector<uint32_t>> classical_clusters = clustering(geometry, events, event_offsets);
  std::vector<std::vector<uint32_t>> cuda_clusters = cuda_clustering_cpu_optimized(geometry, events, event_offsets, verbosity);

  // Statistics about found clusters
  uint32_t found_classical = 0, found_cuda = 0;
  uint32_t cuda_in_classical = 0;

  for (int i=0; i<cuda_clusters.size(); ++i) {
    found_classical += classical_clusters[i].size();
    found_cuda += cuda_clusters[i].size();

    for (auto c : cuda_clusters[i]) {
      if (std::find(classical_clusters[i].begin(), classical_clusters[i].end(), c) != classical_clusters[i].end()) {
        cuda_in_classical++;
      }
    }
  }
  
  std::cout << std::endl << "Classical clustering: " << found_classical << " clusters" << std::endl
    << "Cuda clustering: " << found_cuda << " clusters (" << (100.0 * found_cuda) / ((float) found_classical) << " %)" << std::endl
    << "Cuda clusters in classical: " << (100.0 * cuda_in_classical) / ((float) found_classical) << " %" << std::endl
    << std::endl;



  // std::vector<uint32_t> only_in_cc;
  // std::vector<uint32_t> only_in_cac;
  // for (auto i : cuda_clusters) {
  //   if (std::find(cuda_array_clusters.begin(), cuda_array_clusters.end(), i) == cuda_array_clusters.end()) {
  //     only_in_cc.push_back(i);
  //   }
  // }
  // for (auto i : cuda_array_clusters) {
  //   if (std::find(cuda_clusters.begin(), cuda_clusters.end(), i) == cuda_clusters.end()) {
  //     only_in_cac.push_back(i);
  //   }
  // }

  // std::cout << "Only in cuda_clusters: ";
  // for (auto i : only_in_cc) {
  //   const uint32_t row = (i - 771) / 770;
  //   const uint32_t col = (i - 771) % 770;

  //   std::cout << "(" << i << " " << row << " " << col << ") ";
  // }
  // std::cout << std::endl << std::endl;

  // std::cout << "Only in cuda_array_clusters: ";
  // for (auto i : only_in_cac) {
  //   const uint32_t row = (i - 771) / 770;
  //   const uint32_t col = (i - 771) % 770;

  //   std::cout << "(" << i << " " << row << " " << col << ") ";
  // }
  // std::cout << std::endl;

  // // Set verbosity to max
  // std::cout << std::fixed << std::setprecision(6);
  // logger::ll.verbosityLevel = 3;

  // // Show some statistics
  // statistics(events, event_offsets);

  // // Copy data to pinned host memory
  // char* host_events_pinned;
  // unsigned int* host_event_offsets_pinned;
  // unsigned int* host_hit_offsets_pinned;
  // cudaCheck(cudaMallocHost((void**)&host_events_pinned, events.size()));
  // cudaCheck(cudaMallocHost((void**)&host_event_offsets_pinned, event_offsets.size() * sizeof(unsigned int)));
  // cudaCheck(cudaMallocHost((void**)&host_hit_offsets_pinned, hit_offsets.size() * sizeof(unsigned int)));
  // std::copy_n(std::begin(events), events.size(), host_events_pinned);
  // std::copy_n(std::begin(event_offsets), event_offsets.size(), host_event_offsets_pinned);
  // std::copy_n(std::begin(hit_offsets), hit_offsets.size(), host_hit_offsets_pinned);

  // // Create streams
  // const auto number_of_events = event_offsets.size();
  // std::vector<Stream> streams (tbb_threads);
  // for (int i=0; i<streams.size(); ++i) {
  //   streams[i].initialize(
  //     events,
  //     event_offsets,
  //     hit_offsets,
  //     number_of_events,
  //     events.size(),
  //     transmit_host_to_device,
  //     transmit_device_to_host,
  //     i
  //   );
  // }

  // // Attempt to execute all in one go
  // Timer t;
  // tbb::parallel_for(
  //   static_cast<unsigned int>(0),
  //   static_cast<unsigned int>(tbb_threads),
  //   [&] (unsigned int i) {
  //     auto& s = streams[i];
  //     s(
  //       host_events_pinned,
  //       host_event_offsets_pinned,
  //       host_hit_offsets_pinned,
  //       events.size(),
  //       event_offsets.size(),
  //       hit_offsets.size(),
  //       0,
  //       event_offsets.size(),
  //       number_of_repetitions
  //     );
  //   }
  // );
  // t.stop();

  // std::cout << (event_offsets.size() * tbb_threads * number_of_repetitions / t.get()) << " events/s" << std::endl
  //   << "Ran test for " << t.get() << " seconds" << std::endl;

  // Reset device
  cudaCheck(cudaDeviceReset());

  return 0;
}
