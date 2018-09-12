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
#include "CudaCommon.h"
#include "Logger.h"
#include "Tools.h"
#include "InputTools.h"
#include "Timer.h"
#include "../../stream/sequence/include/StreamWrapper.cuh"
#include "../../stream/sequence/include/InitializeConstants.cuh"

void printUsage(char* argv[]){
  std::cerr << "Usage: "
    << argv[0]
    << std::endl << " -f {folder containing .bin files with VP raw bank information}"
    << std::endl << (mc_check_enabled ? " " : " [") << "-d {folder containing .bin files with MC truth information}"
    << (mc_check_enabled ? "" : " ]")
    << std::endl << " -e {folder containing .bin files with UT hit information}"
    << std::endl << " -i {folder containing .bin files with FT raw bank information}"
    << std::endl << " -g {folder containing geometry descriptions}"
    << std::endl << " -n {number of events to process}=0 (all)"
    << std::endl << " -o {offset of events from which to start}=0 (beginning)"
    << std::endl << " -t {number of threads / streams}=1"
    << std::endl << " -r {number of repetitions per thread / stream}=1"
    << std::endl << " -b {transmit device to host}=1"
    << std::endl << " -c {run checkers}=0"
    << std::endl << " -k {simplified kalman filter}=0"
    << std::endl << " -m {reserve Megabytes}=1024"
    << std::endl << " -v {verbosity}=3 (info)"
    << std::endl << " -p (print memory usage)"
    << std::endl << " -x {run algorithms on x86 architecture as well (if possible)}=0"
    << std::endl;
}

int main(int argc, char *argv[])
{
  std::string folder_name_velopix_raw;
  std::string folder_name_MC = "";
  std::string folder_name_ut_hits = "";
  std::string folder_name_geometry = "";
  std::string folder_name_ft = "";
  uint number_of_files = 0;
  uint start_event_offset = 0;
  uint tbb_threads = 1;
  uint number_of_repetitions = 1;
  uint verbosity = 3;
  bool print_memory_usage = false;
  bool transmit_device_to_host = true;
  // By default, do_check will be true when mc_check is enabled
  bool do_check = mc_check_enabled;
  bool do_simplified_kalman_filter = false;
  bool run_on_x86 = false;
  size_t reserve_mb = 1024;

  signed char c;
  while ((c = getopt(argc, argv, "f:d:e:i:n:o:t:r:pha:b:d:v:c:k:m:g:x:")) != -1) {
    switch (c) {
    case 'f':
      folder_name_velopix_raw = std::string(optarg);
      break;
    case 'd':
      folder_name_MC = std::string(optarg);
      break;
    case 'e':
      folder_name_ut_hits = std::string(optarg);
      break;
    case 'i':
      folder_name_ft = std::string(optarg);
      break;
    case 'g':
      folder_name_geometry = std::string(optarg);
      break;
    case 'm':
      reserve_mb = atoi(optarg);
      break;
    case 'n':
      number_of_files = atoi(optarg);
      break;
    case 'o':
      start_event_offset = atoi(optarg);
      break;
    case 't':
      tbb_threads = atoi(optarg);
      break;
    case 'r':
      number_of_repetitions = atoi(optarg);
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
    case 'x':
      run_on_x86 = atoi(optarg);
      break;
    case 'v':
      verbosity = atoi(optarg);
      break;
    case 'p':
      print_memory_usage = true;
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
  if(folder_name_velopix_raw.empty()){
    std::cerr << "No folder for velopix raw events specified" << std::endl;
    printUsage(argv);
    return -1;
  }
  if(folder_name_ut_hits.empty()){
    std::cerr << "No folder for ut hits specified" << std::endl;
    printUsage(argv);
    return -1;
  }

  if(folder_name_geometry.empty()){
    std::cerr << "No folder for geometry specified" << std::endl;
    printUsage(argv);
    return -1;
  }

  if(folder_name_MC.empty() && mc_check_enabled){
    std::cerr << "No MC folder specified, but MC CHECK turned on" << std::endl;
    printUsage(argv);
    return -1;
  }

  if ( do_check && !mc_check_enabled){
    std::cerr << "Not compiled with -DMC_CHECK=ON, but requesting check" << std::endl;
    return -1;
  }

  // Set verbosity level
  std::cout << std::fixed << std::setprecision(2);
  logger::ll.verbosityLevel = verbosity;

  // Get device properties
  cudaDeviceProp device_properties;
  cudaCheck(cudaGetDeviceProperties(&device_properties, 0));

  // Show call options
  std::cout << "Requested options:" << std::endl
    << " folder with velopix raw bank input (-f): " << folder_name_velopix_raw << std::endl
    << " folder with MC truth input (-d): " << folder_name_MC << std::endl
    << " folder with ut hits input (-e): " << folder_name_ut_hits << std::endl
    << " folder with FT input raw bank input (-i): " << folder_name_ft << std::endl
    << " folder with geometry input (-g): " << folder_name_geometry << std::endl
    << " number of files (-n): " << number_of_files << std::endl
    << " start event offset (-o): " << start_event_offset << std::endl
    << " tbb threads (-t): " << tbb_threads << std::endl
    << " number of repetitions (-r): " << number_of_repetitions << std::endl
    << " transmit device to host (-b): " << transmit_device_to_host << std::endl
    << " run checkers (-c): " << do_check << std::endl
    << " simplified kalman filter (-k): " << do_simplified_kalman_filter << std::endl
    << " reserve MB (-m): " << reserve_mb << std::endl
    << " run algorithms on x86 architecture as well (-x): " << run_on_x86 << std::endl
    << " print memory usage (-p): " << print_memory_usage << std::endl
    << " verbosity (-v): " << verbosity << std::endl
    << " device: " << device_properties.name << std::endl
    << std::endl;

  std::cout << "MC check (compile opt): " << (mc_check_enabled ? "On" : "Off") << std::endl
    << " folder with MC truth input (-d): " << folder_name_MC << std::endl
    << " run checkers (-c): " << do_check << std::endl
    << std::endl;

  if ( do_check && !mc_check_enabled){
    std::cerr << "Not compiled with -DMC_CHECK=ON, but requesting check" << std::endl;
    return -1;
  }

  if ( !do_check && run_on_x86) {
    std::cerr << "Running on x86 only works if MC check is enabled" << std::endl;
    return -1;
  }

  // Read velopix raw data
  std::vector<char> velopix_events;
  std::vector<unsigned int> velopix_event_offsets;
  verbose_cout << "Reading velopix raw events" << std::endl;
  read_folder(
    folder_name_velopix_raw,
    number_of_files,
    velopix_events,
    velopix_event_offsets,
    start_event_offset );

  check_velopix_events( velopix_events, velopix_event_offsets, number_of_files );

  std::string filename_geom = folder_name_geometry + "velo_geometry.bin";
  std::vector<char> velopix_geometry;
  readGeometry(filename_geom, velopix_geometry);

  folder_name_geometry + "ft_geometry.bin";
  std::vector<char> ft_geometry;
  readGeometry(filename_geom, ft_geometry);

  // Copy velopix raw data to pinned host memory
  const int number_of_events = velopix_event_offsets.size() - 1;
  char* host_velopix_events;
  uint* host_velopix_event_offsets;
  cudaCheck(cudaMallocHost((void**)&host_velopix_events, velopix_events.size()));
  cudaCheck(cudaMallocHost((void**)&host_velopix_event_offsets, velopix_event_offsets.size() * sizeof(uint)));
  std::copy_n(std::begin(velopix_events), velopix_events.size(), host_velopix_events);
  std::copy_n(std::begin(velopix_event_offsets), velopix_event_offsets.size(), host_velopix_event_offsets);

  // Read ut hits
  std::vector<char> ut_events;
  std::vector<unsigned int> ut_event_offsets;
  verbose_cout << "Reading UT hits for " << number_of_events << " events " << std::endl;
  read_folder( folder_name_ut_hits, number_of_files,
               ut_events, ut_event_offsets,
               start_event_offset );

  // Copy ut hits to pinned host memory
  VeloUTTracking::HitsSoA* host_ut_hits_events;
  cudaCheck(cudaMallocHost((void**)&host_ut_hits_events, number_of_events * sizeof(VeloUTTracking::HitsSoA)));

  read_ut_events_into_arrays( host_ut_hits_events, ut_events, ut_event_offsets, number_of_events );

  //check_ut_events( host_ut_hits_events, number_of_events );

  // Read LUTs from PrUTMagnetTool into pinned host memory
  PrUTMagnetTool* host_ut_magnet_tool;
  cudaCheck(cudaMallocHost((void**)&host_ut_magnet_tool, sizeof(PrUTMagnetTool)));
  read_UT_magnet_tool( host_ut_magnet_tool );

  //Read and copy FT raw banks
  std::vector<char> ft_events;
  std::vector<uint> ft_event_offsets;
  read_folder(folder_name_ft, number_of_files, ft_events, ft_event_offsets, start_event_offset);

  char* host_ft_events_pinned;
  uint* host_ft_event_offsets_pinned;
  cudaCheck(cudaMallocHost((void**)&host_ft_events_pinned, ft_events.size()));
  cudaCheck(cudaMallocHost((void**)&host_ft_event_offsets_pinned, ft_event_offsets.size() * sizeof(uint)));
  std::copy_n(std::begin(ft_events), ft_events.size(), host_ft_events_pinned);
  std::copy_n(std::begin(ft_event_offsets), ft_event_offsets.size(), host_ft_event_offsets_pinned);

  // Initialize detector constants on GPU
  initializeConstants();

  // Create streams
  StreamWrapper stream_wrapper;
  stream_wrapper.initialize_streams(
    tbb_threads,
    velopix_geometry,
    ft_geometry,
    host_ut_magnet_tool,
    number_of_events,
    transmit_device_to_host,
    do_check,
    do_simplified_kalman_filter,
    print_memory_usage,
    run_on_x86,
    folder_name_MC,
    start_event_offset,
    reserve_mb
  );

  // Attempt to execute all in one go
  Timer t;
  tbb::parallel_for(
    static_cast<uint>(0),
    static_cast<uint>(tbb_threads),
    [&] (uint i) {
      stream_wrapper.run_stream(
        i,
        host_velopix_events,
        host_velopix_event_offsets,
        velopix_events.size(),
        velopix_event_offsets.size(),
        host_ut_hits_events,
        host_ut_magnet_tool,
        host_ft_event_offsets_pinned,
        host_ft_events_pinned,
        ft_event_offsets.size(),
        ft_events.size(),
        number_of_events,
        number_of_repetitions
      );
    }
  );
  t.stop();

  std::cout << (number_of_events * tbb_threads * number_of_repetitions / t.get()) << " events/s" << std::endl
    << "Ran test for " << t.get() << " seconds" << std::endl;

  std::ofstream outfile;
  outfile.open("../tests/test.txt", std::fstream::in | std::fstream::out | std::ios_base::app);
  outfile << start_event_offset << "\t" << (number_of_events * tbb_threads * number_of_repetitions / t.get()) << std::endl;
  outfile.close();

  // Free and reset device
  // cudaCheck(cudaFreeHost(host_velopix_events));
  // cudaCheck(cudaFreeHost(host_velopix_event_offsets));
  cudaCheck(cudaDeviceReset());

  return 0;
}
