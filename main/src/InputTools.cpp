#include "InputTools.h"

/**
 * @brief Test to check existence of filename.
 */
bool exists_test(const std::string& name) {
  std::ifstream f(name.c_str());
  return f.good();
}

/**
 * @brief Natural ordering for strings.
 */
bool naturalOrder(const std::string& s1, const std::string& s2 ) {
  size_t lastindex1 = s1.find_last_of("."); 
  std::string raw1 = s1.substr(0, lastindex1);
  size_t lastindex2 = s2.find_last_of("."); 
  std::string raw2 = s2.substr(0, lastindex2);
  int int1 = stoi(raw1, nullptr, 0);
  int int2 = stoi(raw2, nullptr, 0);
  return int1 < int2;
}

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
 

std::vector<std::string> list_folder(
  const std::string& foldername
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
          filename.find("geometry") == std::string::npos) 
        folderContents.push_back(filename);
    }
    closedir(dir);
    if (folderContents.size() == 0) {
      error_cout << "No binary files found in folder " << foldername << std::endl;
      exit(-1);
    } else {
      verbose_cout << "Found " << folderContents.size() << " binary files" << std::endl;
    }
  } else {
    error_cout << "Folder could not be opened" << std::endl;
    exit(-1);
  }

  // Sort folder contents (file names)
  std::sort(folderContents.begin(), folderContents.end(), naturalOrder);
  return folderContents;
}

/**
 * @brief Reads a number of events from a folder name.
 */
void read_folder(
  const std::string& foldername,
  uint number_of_files,
  std::vector<char>& events,
  std::vector<uint>& event_offsets,
  const uint start_event_offset
) {
  std::vector<std::string> folderContents = list_folder(foldername);

  if (number_of_files == 0) {
    number_of_files = folderContents.size();
  }

  debug_cout << "Requested " << number_of_files << " files" << std::endl;
  int readFiles = 0;

  // Read all requested events
  unsigned int accumulated_size=0;
  std::vector<unsigned int> event_sizes;
  for (int i = start_event_offset; i < number_of_files + start_event_offset; ++i) {
    // Read event #i in the list and add it to the inputs
    std::string readingFile = folderContents[i % folderContents.size()];
    appendFileToVector(foldername + "/" + readingFile, events, event_sizes);

    event_offsets.push_back(accumulated_size);
    accumulated_size += event_sizes.back();
    
    readFiles++;
    if ((readFiles % 100) == 0) {
      info_cout << "." << std::flush;
    }

  }

  // Add last offset
  event_offsets.push_back(accumulated_size);

  debug_cout << std::endl << (event_offsets.size() - 1) << " files read" << std::endl << std::endl;
}

/**
 * @brief Reads the geometry from foldername.
 */
void read_geometry(
  const std::string& filename,
  std::vector<char>& geometry
) {
  if (!exists_test(filename)) {
    throw StrException("Geometry file could not be found: " + filename);
  }
  readFileIntoVector(filename, geometry);
}
