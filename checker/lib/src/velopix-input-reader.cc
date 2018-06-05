/** @file velopix-input-reader.cc
 *
 * @brief reader of velopix input files
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-08
 */

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <dirent.h>

#include "TFile.h"

#include "velopix-input-reader.h"


VelopixEvent::VelopixEvent(const std::vector<uint8_t>& event, const bool checkEvent) {
    uint8_t* input = (uint8_t*) event.data();

    // Event
    numberOfModules  = *((uint32_t*)input); input += sizeof(uint32_t);
    numberOfHits     = *((uint32_t*)input); input += sizeof(uint32_t);
    std::copy_n((float*) input, numberOfModules, std::back_inserter(module_Zs)); input += sizeof(float) * numberOfModules;
    std::copy_n((uint32_t*) input, numberOfModules, std::back_inserter(module_hitStarts)); input += sizeof(uint32_t) * numberOfModules;
    std::copy_n((uint32_t*) input, numberOfModules, std::back_inserter(module_hitNums)); input += sizeof(uint32_t) * numberOfModules;
    std::copy_n((uint32_t*) input, numberOfHits, std::back_inserter(hit_IDs)); input += sizeof(uint32_t) * numberOfHits;
    std::copy_n((float*) input, numberOfHits, std::back_inserter(hit_Xs)); input += sizeof(float) * numberOfHits;
    std::copy_n((float*) input, numberOfHits, std::back_inserter(hit_Ys)); input += sizeof(float) * numberOfHits;
    std::copy_n((float*) input, numberOfHits, std::back_inserter(hit_Zs)); input += sizeof(float) * numberOfHits;

    // Monte Carlo
    uint32_t number_mcp = *((uint32_t*)  input); input += sizeof(uint32_t);
    //std::cout << "n mc particles = " << number_mcp << std::endl;
    for (uint32_t i=0; i<number_mcp; ++i) {
        MCP p;
        p.key          = *((uint32_t*)  input); input += sizeof(uint32_t);
        p.id           = *((uint32_t*)  input); input += sizeof(uint32_t);
        p.p            = *((float*)  input); input += sizeof(float);
        p.pt           = *((float*)  input); input += sizeof(float);
        p.eta          = *((float*)  input); input += sizeof(float);
        p.phi          = *((float*)  input); input += sizeof(float);
        p.islong       = (bool) *((int8_t*)  input); input += sizeof(int8_t);
        p.isdown       = (bool) *((int8_t*)  input); input += sizeof(int8_t);
        p.isvelo       = (bool) *((int8_t*)  input); input += sizeof(int8_t);
        p.isut         = (bool) *((int8_t*)  input); input += sizeof(int8_t);
        p.strangelong  = (bool) *((int8_t*)  input); input += sizeof(int8_t);
        p.strangedown  = (bool) *((int8_t*)  input); input += sizeof(int8_t);
        p.fromb        = (bool) *((int8_t*)  input); input += sizeof(int8_t);
        p.fromd        = (bool) *((int8_t*)  input); input += sizeof(int8_t);
        p.numHits      = *((uint32_t*)  input); input += sizeof(uint32_t);
	//std::cout << "numHits = " << p.numHits << std::endl;
        std::copy_n((uint32_t*) input, p.numHits, std::back_inserter(p.hits)); input += sizeof(uint32_t) * p.numHits;
        mcps.push_back(p);
    }

    size = input - (uint8_t*) event.data();

    if (size != event.size()) {
        throw StrException("Size mismatch in event deserialization: " +
                std::to_string(size) + " vs " + std::to_string(event.size()));
    }

    if (checkEvent) {
        // Check all floats are valid
        for (size_t i=0; i<numberOfModules; ++i) {
            assert(!std::isnan(module_Zs[i]));
            assert(!std::isinf(module_Zs[i]));
        }
        for (size_t i=0; i<numberOfHits; ++i) {
            assert(!std::isnan(hit_Xs[i]));
            assert(!std::isnan(hit_Ys[i]));
            assert(!std::isnan(hit_Zs[i]));
            assert(!std::isinf(hit_Xs[i]));
            assert(!std::isinf(hit_Ys[i]));
            assert(!std::isinf(hit_Zs[i]));
        }
        for (auto& mcp : mcps) {
            assert(!std::isnan(mcp.p));
            assert(!std::isnan(mcp.pt));
            assert(!std::isnan(mcp.eta));
            assert(!std::isnan(mcp.phi));
            assert(!std::isinf(mcp.p));
            assert(!std::isinf(mcp.pt));
            assert(!std::isinf(mcp.eta));
            assert(!std::isinf(mcp.phi));
            // Check all IDs in MC particles exist in hit_IDs
            // for (size_t i=0; i<mcp.numHits; ++i) {
            //     auto hit = mcp.hits[i];
	    // 	//printf("checking hit %u \n", hit );
            //     if (std::find(hit_IDs.begin(), hit_IDs.end(), hit) == hit_IDs.end()) {
            //         throw StrException("The following MC particle hit ID was not found in hit_IDs: " + std::to_string(hit));
            //     }
            // }
        }
        // Check all hitStarts are monotonically increasing (>=)
        // and that they correspond with hitNums
        uint32_t hitStart = 0;
        for (size_t i=0; i<numberOfModules; ++i) {
            if (module_hitNums[i] > 0) {
                if (module_hitStarts[i] < hitStart) {
                    throw StrException("module_hitStarts are not monotonically increasing "
                        + std::to_string(hitStart) + " " + std::to_string(module_hitStarts[i]) + ".");
                }
                hitStart = module_hitStarts[i];
            }
        }
	
    }
}

void VelopixEvent::print() const {
    std::cout << "Event" << std::endl
        << " numberOfModules " << numberOfModules << std::endl
        << " numberOfHits " << numberOfHits << std::endl
        << " module_Zs " << strVector(module_Zs, numberOfModules) << std::endl
        << " module_hitStarts " << strVector(module_hitStarts, numberOfModules) << std::endl
        << " module_hitNums " << strVector(module_hitNums, numberOfModules) << std::endl
        << " hit_IDs " << strVector(hit_IDs, numberOfHits) << std::endl
        << " hit_Xs " << strVector(hit_Xs, numberOfHits) << std::endl
        << " hit_Ys " << strVector(hit_Ys, numberOfHits) << std::endl
        << " hit_Zs " << strVector(hit_Zs, numberOfHits) << std::endl
        << " #MC particles " << mcps.size() << std::endl;

    // Print first MCP
    if (mcps.size() > 0) {
        auto& p = mcps[0];
        std::cout << " First MC particle" << std::endl
            << "  key " << p.key << std::endl
            << "  id " << p.id << std::endl
            << "  p " << p.p << std::endl
            << "  pt " << p.pt << std::endl
            << "  eta " << p.eta << std::endl
            << "  phi " << p.phi << std::endl
            << "  islong " << p.islong << std::endl
            << "  isdown " << p.isdown << std::endl
            << "  isvelo " << p.isvelo << std::endl
            << "  isut " << p.isut << std::endl
            << "  strangelong " << p.strangelong << std::endl
            << "  strangedown " << p.strangedown << std::endl
            << "  fromb " << p.fromb << std::endl
            << "  fromd " << p.fromd << std::endl
            << "  numHits " << p.numHits << std::endl
            << "  hits " << strVector(p.hits, p.numHits) << std::endl;
    }
}

/// get hits into a format we like (should ultimately go away, or just be a view)
VeloPixels VelopixEvent::soaHits() const
{
    VeloPixels retVal;
    retVal.reserve(numberOfHits);
    for (std::size_t i = 0; i < numberOfHits; ++i) {
        retVal.emplace_back(hit_Xs[i], hit_Ys[i], hit_Zs[i], hit_IDs[i]);
    }
    return retVal;
}

MCParticles VelopixEvent::mcparticles() const
{
    using Flags = MCParticleDesc::Flags;
    MCParticles retVal;
    retVal.reserve(mcps.size());
    for (const auto& mcp: mcps) {
	retVal.emplace_back(mcp.p, mcp.pt, mcp.eta, mcp.phi,
		mcp.id, mcp.key, mcp.numHits,
		SomeLHCbIDs(mcp.hits.begin(), mcp.hits.end()),
		(mcp.islong ? Flags::Long : 0) |
		(mcp.isdown ? Flags::Down : 0) |
		(mcp.isvelo ? Flags::Velo : 0) |
		(mcp.isut ? Flags::UT : 0) |
		(mcp.strangelong ? Flags::StrangeLong : 0) |
		(mcp.strangedown ? Flags::StrangeDown : 0) |
		(mcp.fromb ? Flags::FromB : 0) |
		(mcp.fromd ? Flags::FromD : 0));
    }
    return retVal;
}

bool VelopixEventReader::fileExists (const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

void VelopixEventReader::readFileIntoVector(const std::string& filename, std::vector<uint8_t>& output) {
    // Check if file exists
    if (!VelopixEventReader::fileExists(filename)){
        throw StrException("Error: File " + filename + " does not exist.");
    }

    std::ifstream infile(filename.c_str(), std::ifstream::binary);
    infile.seekg(0, std::ios::end);
    auto end = infile.tellg();
    infile.seekg(0, std::ios::beg);
    auto dataSize = end - infile.tellg();
    // read content of infile with a vector
    output.resize(dataSize);
    infile.read((char*) output.data(), dataSize);
    // check that file size and data in file about its payload size match
    const auto currpos = infile.tellg();
    infile.seekg(0, std::ios_base::end);
    const auto endpos = infile.tellg();
    assert(endpos == currpos);
}

bool sortFiles( std::string s1, std::string s2 ) {
  size_t lastindex1 = s1.find_last_of("."); 
  std::string raw1 = s1.substr(0, lastindex1);
  size_t lastindex2 = s2.find_last_of("."); 
  std::string raw2 = s2.substr(0, lastindex2);
  int int1 = stoi(raw1, nullptr, 0);
  int int2 = stoi(raw2, nullptr, 0);
  return int1 < int2;
}

std::vector< std::string > VelopixEventReader::getFolderContents(
  const std::string& foldername,
  uint nFiles
  ) {
  
  std::vector<std::string> folderContents;
  DIR *dir;
  struct dirent *ent;
  
  if ((dir = opendir(foldername.c_str())) != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir(dir)) != NULL) {
      std::string filename = std::string(ent->d_name);
      if (filename.find(".bin") != std::string::npos) {
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
  
  std::sort( folderContents.begin(), folderContents.end(), sortFiles );
 
  return folderContents;
}
  
std::vector<VelopixEvent> VelopixEventReader::readFolder (
        const std::string& foldername,
        uint nFiles,
        const bool checkEvents
        ) {

  std::vector< std::string > folderContents = getFolderContents( foldername, nFiles );

  uint requestedFiles = nFiles==0 ? folderContents.size() : nFiles;
  std::cout << "Requested " << requestedFiles << " files" << std::endl;
  
  if ( requestedFiles > folderContents.size() ) {
    std::cout << "ERROR: requested " << requestedFiles << " files, but only " << folderContents.size() << " files are present" << std::endl;
    exit(-1);
  }
   
  std::vector<VelopixEvent> input;
  int readFiles = 0;
  for (uint i=0; i<requestedFiles; ++i) {
    // Read event #i in the list and add it to the inputs
    // if more files are requested than present in folder, read them again
    std::string readingFile = folderContents[i % folderContents.size()];
    std::cout << "Reading MC event " << readingFile << std::endl;
    
    std::vector<uint8_t> inputContents;
    readFileIntoVector(foldername + "/" + readingFile, inputContents);
    //std::cout << "vector size = " << inputContents.size() << std::endl;
    
    // Check the number of sensors is correct, otherwise ignore it
    VelopixEvent event {inputContents, checkEvents};
    if ( i == 0 )
      event.print();
    
    // Sanity check
    if (event.numberOfModules == VelopixEventReader::numberOfModules) {
      input.emplace_back(event);
    }
    else
      printf("ERROR: number of sensors should be %u, but it is %u \n", VelopixEventReader::numberOfModules, event.numberOfModules);  
    
    
        readFiles++;
        if ((readFiles % 100) == 0) {
	  std::cout << "." << std::flush;
        }
	
  }
  
  std::cout << std::endl << input.size() << " files read" << std::endl << std::endl;
  return input;
}

std::vector<VelopixEvent> VelopixEventReader::get_mcps_from_ntuple( const std::string& foldername, uint nFiles ) {

  std::vector<VelopixEvent> events;

  /* Read input ROOT file */
  //TFile *f = new TFile();
  
  return events;
}

		     
