#pragma once

#include <dirent.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <map>
#include <cmath>
#include <stdint.h>
#include "Logger.h"

bool exists_test(
  const std::string& name
);

bool naturalOrder(
  const std::string& s1,
  const std::string& s2
);

void readFileIntoVector(
  const std::string& filename,
  std::vector<char>& events
);

void appendFileToVector(
  const std::string& filename,
  std::vector<char>& events,
  std::vector<unsigned int>& event_sizes
);

std::vector<std::string> list_folder(
  const std::string& foldername,
  const bool fromNtuple = false
);
 
void read_folder(
  const std::string& foldername,
  unsigned int fileNumber,
  std::vector<char>& events,
  std::vector<unsigned int>& event_offsets,
  const uint start_event_offset
);
