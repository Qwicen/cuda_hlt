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
  // for ( int i_file = 0; i_file < folderContents.size(); ++i_file) {
  //   debug_cout << "file " << i_file << " is called " << folderContents[i_file] << std::endl;
  //}

  return folderContents;
}

/**
 * @brief Processes the number of events requested. If 0,
 *        returns the size of the passed folder contents.
 */
uint get_number_of_events_requested(
  uint number_of_events_requested,
  const std::string& foldername
) {
  if (number_of_events_requested > 0) {
    return number_of_events_requested;
  } else {
    std::vector<std::string> folderContents = list_folder(foldername);
    return folderContents.size();
  }
}

/**
 * @brief Reads a number of events from a folder name.
 */
void read_folder(
  const std::string& foldername,
  uint number_of_events_requested,
  std::vector<char>& events,
  std::vector<uint>& event_offsets,
  const uint start_event_offset
) {
  std::vector<std::string> folderContents = list_folder(foldername);

  debug_cout << "Requested " << number_of_events_requested << " files" << std::endl;
  int readFiles = 0;

  // Read all requested events
  unsigned int accumulated_size=0;
  std::vector<unsigned int> event_sizes;
  for (int i = start_event_offset; i < number_of_events_requested + start_event_offset; ++i) {
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

/**
 * @brief Reads the UT magnet tool.
 */
void read_UT_magnet_tool(
  const std::string& folder_name,
  std::vector<char>& ut_magnet_tool
) {
  ut_magnet_tool.resize(sizeof(PrUTMagnetTool));
  PrUTMagnetTool* pr_ut_magnet_tool = (PrUTMagnetTool*) ut_magnet_tool.data();

  //load the deflection and Bdl values from a text file
  std::ifstream deflectionfile;
  std::string filename = folder_name + "/deflection.txt";
  if (!exists_test(filename)) {
    throw StrException("Deflection table file could not be found: " + filename);
  }
  deflectionfile.open(filename);
  if (deflectionfile.is_open()) {
    int i = 0;
    float deflection;
    while (!deflectionfile.eof()) {
      deflectionfile >> deflection;
      assert( i < PrUTMagnetTool::N_dxLay_vals );
      pr_ut_magnet_tool->dxLayTable[i++] = deflection;
    }
  }
  
  std::ifstream bdlfile;
  filename = folder_name + "/bdl.txt";
  if (!exists_test(filename)) {
    throw StrException("Bdl table file could not be found: " + filename);
  }
  bdlfile.open(filename);
  if (bdlfile.is_open()) {
    int i = 0;
    float bdl;
    while (!bdlfile.eof()) {
      bdlfile >> bdl;
      assert( i < PrUTMagnetTool::N_bdl_vals );
      pr_ut_magnet_tool->bdlTable[i++] = bdl;
    }
  }
}
