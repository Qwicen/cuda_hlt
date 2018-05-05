#include "../include/Tools.h"

/**
 * @brief Read files into vectors.
 */
void readFileIntoVector(
  const std::string& filename,
  std::vector<char>& events
) {
  std::ifstream infile(filename.c_str(), std::ifstream::binary);
  infile.seekg(0, std::ios::end);
  auto end = infile.tellg();
  infile.seekg(0, std::ios::beg);
  auto dataSize = end - infile.tellg();

  events.resize(dataSize);
  infile.read((char*) &(events[0]), dataSize);
  infile.close();
}

/**
 * @brief Appends a file to a vector of char.
 *        It can also be used to read a file to a vector,
 *        returning the event_size.
 */
void appendFileToVector(
  const std::string& filename,
  std::vector<char>& events,
  std::vector<unsigned int>& event_sizes
) {
  std::ifstream infile(filename.c_str(), std::ifstream::binary);
  infile.seekg(0, std::ios::end);
  auto end = infile.tellg();
  infile.seekg(0, std::ios::beg);
  auto dataSize = end - infile.tellg();

  // read content of infile with a vector
  const size_t previous_size = events.size();
  events.resize(events.size() + dataSize);
  infile.read(events.data() + previous_size, dataSize);
  event_sizes.push_back(dataSize);
  infile.close();
}

void readGeometry(
  const std::string& foldername,
  std::vector<char>& geometry
) {
  readFileIntoVector(foldername + "/geometry.bin", geometry);
}

/**
 * @brief Reads a number of events from a folder name.
 */
void readFolder(
  const std::string& foldername,
  unsigned int number_of_files,
  std::vector<char>& events,
  std::vector<unsigned int>& event_offsets
) {
  std::vector<std::string> folderContents;
  DIR *dir;
  struct dirent *ent;

  // Find out folder contents
  if ((dir = opendir(foldername.c_str())) != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir(dir)) != NULL) {
      std::string filename = std::string(ent->d_name);
      if (filename.find(".bin") != std::string::npos &&
        filename.find("geometry") == std::string::npos) {
        folderContents.push_back(filename);
      }
    }
    closedir(dir);
    if (folderContents.size() == 0) {
      std::cerr << "No binary files found in folder " << foldername << std::endl;
      exit(-1);
    } else {
      std::cout << "Found " << folderContents.size() << " binary files" << std::endl;
    }
  } else {
    std::cerr << "Folder could not be opened" << std::endl;
    exit(-1);
  }

  // Sort folder contents
  std::sort(folderContents.begin(), folderContents.end()); 

  if (number_of_files == 0) {
    number_of_files = folderContents.size();
  }

  std::cout << "Requested " << number_of_files << " files" << std::endl;
  int readFiles = 0;

  // Read all requested events
  unsigned int accumulated_size=0;
  std::vector<unsigned int> event_sizes;
  for (int i=0; i<number_of_files; ++i) {
    // Read event #i in the list and add it to the inputs
    std::string readingFile = folderContents[i % folderContents.size()];
    appendFileToVector(foldername + "/" + readingFile, events, event_sizes);

    event_offsets.push_back(accumulated_size);
    accumulated_size += event_sizes.back();

    readFiles++;
    if ((readFiles % 100) == 0) {
      std::cout << "." << std::flush;
    }
  }

  // Add last offset
  event_offsets.push_back(accumulated_size);

  std::cout << std::endl << (event_offsets.size() - 1) << " files read" << std::endl << std::endl;
}

/**
 * @brief Print statistics from the input files
 */
void statistics(
  const std::vector<char>& input,
  std::vector<unsigned int>& event_offsets
) {
  // unsigned int max_number_of_hits = 0;
  // unsigned int max_number_of_hits_in_module = 0;
  // unsigned int average_number_of_hits_in_module = 0;

  // for (size_t i=0; i<event_offsets.size(); ++i) {
  //   EventInfo info (input.data() + event_offsets[i]);
  //   for (size_t j=0; j<info.numberOfModules; ++j) {
  //     max_number_of_hits_in_module = std::max(max_number_of_hits_in_module, info.module_hitNums[j]);
  //     average_number_of_hits_in_module += info.module_hitNums[j];
  //   }
  //   max_number_of_hits = std::max(max_number_of_hits, info.numberOfHits);
  // }
  // average_number_of_hits_in_module /= event_offsets.size() * 52;

  // std::cout << "Statistics on input events:" << std::endl
  //   << " Max number of hits in event: " << max_number_of_hits << std::endl
  //   << " Max number of hits in one module: " << max_number_of_hits_in_module << std::endl
  //   << " Average number of hits in module: " << average_number_of_hits_in_module << std::endl << std::endl;
}

/**
 * @brief Obtains results statistics.
 */
std::map<std::string, float> calcResults(std::vector<float>& times){
    // sqrt ( E( (X - m)2) )
    std::map<std::string, float> results;
    float deviation = 0.0f, variance = 0.0f, mean = 0.0f, min = FLT_MAX, max = 0.0f;

    for(auto it = times.begin(); it != times.end(); it++){
        const float seconds = (*it);
        mean += seconds;
        variance += seconds * seconds;

        if (seconds < min) min = seconds;
        if (seconds > max) max = seconds;
    }

    mean /= times.size();
    variance = (variance / times.size()) - (mean * mean);
    deviation = std::sqrt(variance);

    results["variance"] = variance;
    results["deviation"] = deviation;
    results["mean"] = mean;
    results["min"] = min;
    results["max"] = max;

    return results;
}
