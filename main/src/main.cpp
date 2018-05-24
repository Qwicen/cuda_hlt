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
    << std::endl << " -f {folder containing .bin files}"
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
  std::string folder_name;
  unsigned int number_of_files = 0;
  unsigned int tbb_threads = 3;
  unsigned int number_of_repetitions = 10;
  unsigned int verbosity = 3;
  bool print_individual_rates = false;
  bool transmit_host_to_device = true;
  bool transmit_device_to_host = true;
  bool do_check = false;
  bool do_simplified_kalman_filter = false;
   
  signed char c;
  while ((c = getopt(argc, argv, "f:n:t:r:pha:b:d:v:c:k:")) != -1) {
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

  if ( do_mc_check ) 
    printf("MC check ON \n");
  else
    printf("MC check OFF \n");
  
  // Check how many files were specified and
  // call the entrypoint with the suggested format
  if(folder_name.empty()){
    std::cerr << "No folder specified" << std::endl;
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
    << " folder (-f): " << folder_name << std::endl
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
  std::vector<char> events;
  std::vector<unsigned int> event_offsets;
  readFolder(folder_name, number_of_files, events, event_offsets);

  std::vector<char> geometry;
  readGeometry(folder_name, geometry);

  // Copy data to pinned host memory
  char* host_events_pinned;
  unsigned int* host_event_offsets_pinned;
  cudaCheck(cudaMallocHost((void**)&host_events_pinned, events.size()));
  cudaCheck(cudaMallocHost((void**)&host_event_offsets_pinned, event_offsets.size() * sizeof(unsigned int)));
  std::copy_n(std::begin(events), events.size(), host_events_pinned);
  std::copy_n(std::begin(event_offsets), event_offsets.size(), host_event_offsets_pinned);

  // // Call clustering
  // std::vector<std::vector<uint32_t>> clusters = cuda_clustering_simplified(
  //   geometry,
  //   events,
  //   event_offsets,
  //   true
  // );

  // std::vector<std::vector<uint32_t>> clusters_simplified = cuda_clustering_cpu_optimized(
  //   geometry,
  //   events,
  //   event_offsets,
  //   true
  // );

  // std::vector<std::vector<uint32_t>> clusters_classical = clustering(
  //   geometry,
  //   events,
  //   event_offsets,
  //   true
  // );

  // uint found_clusters = 0;
  // uint found_clusters_classical = 0;
  // uint found_clusters_simplified = 0;

  // for (int i=0; i<clusters.size(); ++i) {
  //   found_clusters += clusters[i].size();
  //   found_clusters_simplified += clusters_simplified[i].size();
  //   found_clusters_classical += clusters_classical[i].size();
  // }

  // std::cout << "Found classical: " << found_clusters_classical << std::endl
  //   << "Found cuda simplified: " << found_clusters << " (" << (100.f * (((float) found_clusters) / ((float) found_clusters_classical))) << " %)" << std::endl
  //   << "Found cuda simplified cpu optimized: " << found_clusters_simplified << " (" << (100.f * (((float) found_clusters_simplified) / ((float) found_clusters_classical))) << " %)" << std::endl;

  // auto cluster_sum = 0;
  // std::cout << "Reconstructed clusters:" << std::endl;
  // for (int i=0; i<clusters.size(); ++i) {
  //   std::cout << i << ": " << clusters[i].size() << std::endl;
  //   cluster_sum += clusters[i].size();
  // }
  // std::cout << "Reconstructed cluster total: " << cluster_sum << std::endl;

  // cluster_sum = 0;
  // std::cout << "Reconstructed clusters (simplified):" << std::endl;
  // for (int i=0; i<clusters_simplified.size(); ++i) {
  //   std::cout << i << ": " << clusters_simplified[i].size() << std::endl;
  //   cluster_sum += clusters_simplified[i].size();
  // }
  // std::cout << "Reconstructed cluster total (simplified): " << cluster_sum << std::endl;

  // cluster_sum = 0;
  // std::cout << "Reconstructed clusters (classical):" << std::endl;
  // for (int i=0; i<clusters_classical.size(); ++i) {
  //   std::cout << i << ": " << clusters_classical[i].size() << std::endl;
  //   cluster_sum += clusters_classical[i].size();
  // }
  // std::cout << "Reconstructed cluster total (classical): " << cluster_sum << std::endl;

  // Initialize detector constants on GPU
  initializeConstants();

  // Create streams
  const auto number_of_events = event_offsets.size() - 1;
  std::vector<Stream> streams (tbb_threads);
  for (int i=0; i<streams.size(); ++i) {
    streams[i].initialize(
      events,
      event_offsets,
      geometry,
      number_of_events,
      events.size(),
      transmit_host_to_device,
      transmit_device_to_host,
      do_check,
      do_simplified_kalman_filter,
      print_individual_rates,
      i
    );
    // Memory consumption
    size_t free_byte ;
    size_t total_byte ;
    cudaCheck( cudaMemGetInfo( &free_byte, &total_byte ) );
    float free_percent = (float)free_byte / total_byte * 100;
    float used_percent = (float)(total_byte - free_byte) / total_byte * 100;
    printf("GPU memory: %f percent free, %f percent used \n", free_percent, used_percent );
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
        events.size(),
        event_offsets.size(),
        0,
        number_of_events,
        number_of_repetitions
      );
    }
  );
  t.stop();

  std::cout << (number_of_events * tbb_threads * number_of_repetitions / t.get()) << " events/s" << std::endl
    << "Ran test for " << t.get() << " seconds" << std::endl;

  // // Memory consumption
  // size_t free_byte ;
  // size_t total_byte ;
  // cudaCheck( cudaMemGetInfo( &free_byte, &total_byte ) );
  // float free_percent = (float)free_byte / total_byte * 100;
  // float used_percent = (float)(total_byte - free_byte) / total_byte * 100;
  // printf("GPU memory: %f percent free, %f percent used \n", free_percent, used_percent );
  
  // Free and reset device
  cudaCheck(cudaFreeHost(host_events_pinned));
  cudaCheck(cudaFreeHost(host_event_offsets_pinned));
  cudaCheck(cudaDeviceReset());

  return 0;
}
