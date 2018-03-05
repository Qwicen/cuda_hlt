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
#include "tbb/tbb.h"
#include "cuda_runtime.h"
#include "../include/Common.h"
#include "../include/Logger.h"
#include "../include/Tools.h"
#include "../../stream/include/Stream.cuh"
#include "../include/Timer.h"

void printUsage(char* argv[]){
  std::cerr << "Usage: "
    << argv[0] << " <folder containing .bin files> <number of files to process>"
    << std::endl;
}

int main(int argc, char *argv[])
{
  std::string foldername;
  std::vector<std::string> folderContents;
  int fileNumber;
  unsigned int tbb_threads = 4;

  // Get params (getopt independent - Compatible with Windows)
  if (argc < 3){
    printUsage(argv);
    return 0;
  }
  foldername = std::string(argv[1]);
  fileNumber = atoi(argv[2]);

  // Check how many files were specified and
  // call the entrypoint with the suggested format
  if(foldername.empty()){
    std::cerr << "No folder specified" << std::endl;
    printUsage(argv);
    return -1;
  }

  // Read folder contents
  const std::vector<std::vector<unsigned char>> input = readFolder(foldername, fileNumber);

  // Call offloaded algo
  std::vector<std::vector<unsigned char>> output;

  // Set verbosity to max
  std::cout << std::fixed << std::setprecision(6);
  logger::ll.verbosityLevel = 3;

  // Show some statistics
  statistics(input);

  // Attempt to execute all in one go
  Timer t;
  t.start();
  tbb::parallel_for(
    static_cast<unsigned int>(0),
    static_cast<unsigned int>(tbb_threads),
    [&] (unsigned int i) {
      Stream s (i);
      s(input, 0, input.size());
    }
  );
  t.stop();

  std::cout << (input.size() * tbb_threads / t.get()) << " events/s combined" << std::endl;

  // Reset device
  cudaCheck(cudaDeviceReset());

  return 0;
}
