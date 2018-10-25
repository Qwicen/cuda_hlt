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
#include "RuntimeOptions.h"
#include "Logger.h"
#include "Tools.h"
#include "InputTools.h"
#include "InputReader.h"
#include "Timer.h"
#include "StreamWrapper.cuh"
#include "Constants.cuh"

void printUsage(char* argv[]){
  std::cerr << "Usage: "
    << argv[0]
    << std::endl << " -f {folder containing directories with raw bank binaries for every sub-detector}"
    << std::endl << " -g {folder containing detector configuration}"
    << std::endl << " -d {folder containing .bin files with MC truth information}"
    << std::endl << " -n {number of events to process}=0 (all)"
    << std::endl << " -o {offset of events from which to start}=0 (beginning)"
    << std::endl << " -t {number of threads / streams}=1"
    << std::endl << " -r {number of repetitions per thread / stream}=1"
    << std::endl << " -c {run checkers}=0"
    << std::endl << " -k {simplified kalman filter}=0"
    << std::endl << " -m {reserve Megabytes}=1024"
    << std::endl << " -v {verbosity}=3 (info)"
    << std::endl << " -p {print memory usage}=0"
    << std::endl << " -a {run only data preparation algorithms: decoding, clustering, sorting}=0"
    << std::endl << " -x {run algorithms on x86 architecture if implementation is available}=0"
    << std::endl;
}

int main(int argc, char *argv[])
{
  std::string folder_name_raw_banks = "../input/minbias/banks/";
  std::string folder_name_MC = "../input/minbias/MC_info/";
  std::string folder_name_detector_configuration = "../input/detector_configuration/";
  uint number_of_events_requested = 0;
  uint start_event_offset = 0;
  uint tbb_threads = 1;
  uint number_of_repetitions = 1;
  uint verbosity = 3;
  bool print_memory_usage = false;
  // By default, do_check will be true when mc_check is enabled
  bool do_check = true;
  bool do_simplified_kalman_filter = false;
  bool run_on_x86 = false;
  size_t reserve_mb = 1024;

  signed char c;
  while ((c = getopt(argc, argv, "f:d:n:o:t:r:pha:b:d:v:c:k:m:g:x")) != -1) {
    switch (c) {
    case 'f':
      folder_name_raw_banks = std::string(optarg);
      break;
    case 'd':
      folder_name_MC = std::string(optarg);
      break;
    case 'g':
      folder_name_detector_configuration = std::string(optarg);
      break;
    case 'm':
      reserve_mb = atoi(optarg);
      break;
    case 'n':
      number_of_events_requested = atoi(optarg);
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

  // Options sanity check
  if (folder_name_raw_banks.empty() || folder_name_detector_configuration.empty() || (folder_name_MC.empty() && do_check)) {
    std::string missing_folder = "";

    if (folder_name_raw_banks.empty()) missing_folder = "raw banks";
    else if (folder_name_detector_configuration.empty()) missing_folder = "detector geometry";
    else if (folder_name_MC.empty() && do_check) missing_folder = "Monte Carlo";

    error_cout << "No folder for " << missing_folder << " specified" << std::endl;
    printUsage(argv);
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
    << " folder containing directories with raw bank binaries for every sub-detector (-f): " << folder_name_raw_banks << std::endl
    << " folder with detector configuration (-g): " << folder_name_detector_configuration << std::endl
    << " folder with MC truth input (-d): " << folder_name_MC << std::endl
    << " run checkers (-c): " << do_check << std::endl
    << " number of files (-n): " << number_of_events_requested << std::endl
    << " start event offset (-o): " << start_event_offset << std::endl
    << " tbb threads (-t): " << tbb_threads << std::endl
    << " number of repetitions (-r): " << number_of_repetitions << std::endl
    << " simplified kalman filter (-k): " << do_simplified_kalman_filter << std::endl
    << " reserve MB (-m): " << reserve_mb << std::endl
    << " run algorithms on x86 architecture if implementation is available (-x): " << run_on_x86 << std::endl
    << " print memory usage (-p): " << print_memory_usage << std::endl
    << " verbosity (-v): " << verbosity << std::endl
    << " device: " << device_properties.name << std::endl
    << std::endl;

  // Read all inputs
  info_cout << "Reading input datatypes" << std::endl;

  std::string folder_name_velopix_raw = folder_name_raw_banks + "VP"; 
  number_of_events_requested = get_number_of_events_requested(
    number_of_events_requested, folder_name_velopix_raw);

  std::string folder_name_UT_raw = folder_name_raw_banks + "UT";
  std::string folder_name_SciFi_raw = folder_name_raw_banks + "FTCluster";
  auto geometry_reader = GeometryReader(folder_name_detector_configuration);
  auto ut_magnet_tool_reader = UTMagnetToolReader(folder_name_detector_configuration);
  auto velo_reader = VeloReader(folder_name_velopix_raw);
  auto ut_reader = EventReader(folder_name_UT_raw);
  auto scifi_reader = EventReader(folder_name_SciFi_raw);

  std::vector<char> velo_geometry = geometry_reader.read_geometry("velo_geometry.bin");
  std::vector<char> ut_boards = geometry_reader.read_geometry("ut_boards.bin");
  std::vector<char> ut_geometry = geometry_reader.read_geometry("ut_geometry.bin");
  std::vector<char> ut_magnet_tool = ut_magnet_tool_reader.read_UT_magnet_tool();
  std::vector<char> scifi_geometry = geometry_reader.read_geometry("scifi_geometry.bin");
  velo_reader.read_events(number_of_events_requested, start_event_offset);
  ut_reader.read_events(number_of_events_requested, start_event_offset);
  scifi_reader.read_events(number_of_events_requested, start_event_offset);

  info_cout << std::endl << "All input datatypes successfully read" << std::endl << std::endl;

  // Initialize detector constants on GPU
  Constants constants;
  constants.reserve_and_initialize();
  constants.initialize_ut_decoding_constants(ut_geometry);
  constants.initialize_geometry_constants(
    velo_geometry,
    ut_boards,
    ut_geometry,
    ut_magnet_tool,
    scifi_geometry);

  // Create streams
  StreamWrapper stream_wrapper;
  stream_wrapper.initialize_streams(
    tbb_threads,
    number_of_events_requested,
    do_check,
    do_simplified_kalman_filter,
    print_memory_usage,
    run_on_x86,
    folder_name_MC,
    start_event_offset,
    reserve_mb,
    constants
  );

  // Attempt to execute all in one go
  Timer t;
  tbb::parallel_for(
    static_cast<uint>(0),
    static_cast<uint>(tbb_threads),
    [&] (uint i) {
      auto runtime_options = RuntimeOptions{
        velo_reader.host_events,
        velo_reader.host_event_offsets,
        velo_reader.host_events_size,
        velo_reader.host_event_offsets_size,
        ut_reader.host_events,
        ut_reader.host_event_offsets,
        ut_reader.host_events_size,
        ut_reader.host_event_offsets_size,
        scifi_reader.host_events,
        scifi_reader.host_event_offsets,
        scifi_reader.host_events_size,
        scifi_reader.host_event_offsets_size,
        number_of_events_requested,
        number_of_repetitions};

      stream_wrapper.run_stream(i, runtime_options);
    }
  );
  t.stop();

  // Do optional Monte Carlo truth test
  if (do_check) {
    stream_wrapper.run_monte_carlo_test(0, number_of_events_requested);
  }

  std::cout << (number_of_events_requested * tbb_threads * number_of_repetitions / t.get()) << " events/s" << std::endl
    << "Ran test for " << t.get() << " seconds" << std::endl;

  // Reset device
  cudaCheck(cudaDeviceReset());

  return 0;
}
