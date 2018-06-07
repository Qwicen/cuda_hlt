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

// for Ntuple reading
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TChain.h"

#include "velopix-input-reader.h"


VelopixEvent::VelopixEvent(const std::vector<uint8_t>& event, const bool checkEvent) {
  uint8_t* input = (uint8_t*) event.data();

  // Event
  // numberOfModules  = *((uint32_t*)input); input += sizeof(uint32_t);
  // numberOfHits   = *((uint32_t*)input); input += sizeof(uint32_t);
  // std::copy_n((float*) input, numberOfModules, std::back_inserter(module_Zs)); input += sizeof(float) * numberOfModules;
  // std::copy_n((uint32_t*) input, numberOfModules, std::back_inserter(module_hitStarts)); input += sizeof(uint32_t) * numberOfModules;
  // std::copy_n((uint32_t*) input, numberOfModules, std::back_inserter(module_hitNums)); input += sizeof(uint32_t) * numberOfModules;
  // std::copy_n((uint32_t*) input, numberOfHits, std::back_inserter(hit_IDs)); input += sizeof(uint32_t) * numberOfHits;
  // std::copy_n((float*) input, numberOfHits, std::back_inserter(hit_Xs)); input += sizeof(float) * numberOfHits;
  // std::copy_n((float*) input, numberOfHits, std::back_inserter(hit_Ys)); input += sizeof(float) * numberOfHits;
  // std::copy_n((float*) input, numberOfHits, std::back_inserter(hit_Zs)); input += sizeof(float) * numberOfHits;

  // Monte Carlo
  uint32_t number_mcp = *((uint32_t*)  input); input += sizeof(uint32_t);
  for (uint32_t i=0; i<number_mcp; ++i) {
    MCP p;
    p.key      = *((uint32_t*)  input); input += sizeof(uint32_t);
    p.id       = *((uint32_t*)  input); input += sizeof(uint32_t);
    p.p      = *((float*)  input); input += sizeof(float);
    p.pt       = *((float*)  input); input += sizeof(float);
    p.eta      = *((float*)  input); input += sizeof(float);
    p.phi      = *((float*)  input); input += sizeof(float);
    p.islong     = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.isdown     = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.isvelo     = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.isut     = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.strangelong  = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.strangedown  = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.fromb    = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.fromd    = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.numHits    = *((uint32_t*)  input); input += sizeof(uint32_t);
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
    }
    
    // Check all hitStarts are monotonically increasing (>=)
    // and that they correspond with hitNums
    // uint32_t hitStart = 0;
    // for (size_t i=0; i<numberOfModules; ++i) {
    //   if (module_hitNums[i] > 0) {
    //     if (module_hitStarts[i] < hitStart) {
    //       throw StrException("module_hitStarts are not monotonically increasing "
    // 			     + std::to_string(hitStart) + " " + std::to_string(module_hitStarts[i]) + ".");
    //     }
    //     hitStart = module_hitStarts[i];
    //   }
    // }

}

void VelopixEvent::print() const {
  std::cout << "Event" << std::endl
        // << " numberOfModules " << numberOfModules << std::endl
        // << " numberOfHits " << numberOfHits << std::endl
        // << " module_Zs " << strVector(module_Zs, numberOfModules) << std::endl
        // << " module_hitStarts " << strVector(module_hitStarts, numberOfModules) << std::endl
        // << " module_hitNums " << strVector(module_hitNums, numberOfModules) << std::endl
        // << " hit_IDs " << strVector(hit_IDs, numberOfHits) << std::endl
        // << " hit_Xs " << strVector(hit_Xs, numberOfHits) << std::endl
        // << " hit_Ys " << strVector(hit_Ys, numberOfHits) << std::endl
        // << " hit_Zs " << strVector(hit_Zs, numberOfHits) << std::endl
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
  const bool fromNtuple,
  uint nFiles
  ) {
  
  std::vector<std::string> folderContents;
  DIR *dir;
  struct dirent *ent;
  
  if ((dir = opendir(foldername.c_str())) != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir(dir)) != NULL) {
      std::string filename = std::string(ent->d_name);
      if ( !fromNtuple ) {
	if (filename.find(".bin") != std::string::npos) {
	  folderContents.push_back(filename);
	}
      }
      else if ( fromNtuple ) {
	if (filename.find(".root") != std::string::npos) {
	  folderContents.push_back(filename);
	}
      }
    }
    closedir(dir);
    if (folderContents.size() == 0) {
      if ( fromNtuple )
	std::cerr << "No root files found in folder " << foldername << std::endl;
      else if ( !fromNtuple )
	std::cerr << "No binary files found in folder " << foldername << std::endl;
      exit(-1);
    } else {
      info_cout << "Found " << folderContents.size() << " binary files" << std::endl;
    }
  } else {
    std::cerr << "Folder could not be opened" << std::endl;
    exit(-1);
  }
  
  std::sort( folderContents.begin(), folderContents.end(), sortFiles );
 
  return folderContents;
}

/* Use Ntuple created by PrTrackerDumper tool, 
   written by Renato Quagliani,
   Use code from https://gitlab.cern.ch/rquaglia/PrOfflineStudies
   by Renato Quagliani to read the input
*/
void VelopixEventReader::readNtupleIntoVelopixEvent(
  const std::string& filename,
  const std::string& trackType,
  VelopixEvent& event							    
 ) {

  // Check if file exists
  if (!VelopixEventReader::fileExists(filename)){
    throw StrException("Error: File " + filename + " does not exist.");
  }
  
  TFile file(filename.data(),"READ");
  TTree *tree = (TTree*)file.Get("Hits_detectors");
  assert( tree);

  // MCP information
  TBranch        *b_fullInfo;   //!
  TBranch        *b_hasSciFi;   //!
  TBranch        *b_hasUT;   //!
  TBranch        *b_hasVelo;   //!
  TBranch        *b_isDown;   //!
  TBranch        *b_isDown_noVelo;   //!
  TBranch        *b_isLong;   //!
  TBranch        *b_key;   //!
  TBranch        *b_isLong_andUT;   //!
  TBranch        *b_p;   //!
  TBranch        *b_pt;   //!
  TBranch        *b_pid;   //!
  TBranch        *b_eta;   //!
  TBranch        *b_ovtx_x;   //!
  TBranch        *b_ovtx_y;   //!
  TBranch        *b_ovtx_z;   //!
  TBranch        *b_fromBeautyDecay;   //!
  TBranch        *b_fromCharmDecay;   //!
  TBranch        *b_fromStrangeDecay;   //!
  TBranch        *b_DecayOriginMother_pid;   //!

  // LHCbIDs of hits in Velo
  TBranch        *b_Velo_lhcbID;   //!
  // LHCbIDs of hits in UT
  TBranch        *b_UT_lhcbID;   //!
  //LHCbIDs of hits in SciFi (FT)
  TBranch        *b_FT_lhcbID;   //!
  
  TTree* fChain = tree;

  // Declare variables to store Tree variables in
  std::vector<unsigned int> *Velo_lhcbID;
  std::vector<unsigned int> *FT_lhcbID;
  std::vector<unsigned int> *UT_lhcbID;

  Bool_t          fullInfo;
  Bool_t          hasSciFi;
  Bool_t          hasUT;
  Bool_t          hasVelo;
  Bool_t          isDown;
  Bool_t          isDown_noVelo;
  Bool_t          isLong;
  Int_t           key;
  Bool_t          isLong_andUT;
  Double_t        p;
  Double_t        pt;
  Int_t           pid;
  Double_t        eta;
  Double_t        ovtx_x;
  Double_t        ovtx_y;
  Double_t        ovtx_z;
  Bool_t          fromBeautyDecay;
  Bool_t          fromCharmDecay;
  Bool_t          fromStrangeDecay;
  Int_t           DecayOriginMother_pid;
  
  // Set branch addresses and branch pointers
  fChain->SetBranchAddress("fullInfo", &fullInfo, &b_fullInfo);
  fChain->SetBranchAddress("hasSciFi", &hasSciFi, &b_hasSciFi);
  fChain->SetBranchAddress("hasUT", &hasUT, &b_hasUT);
  fChain->SetBranchAddress("hasVelo", &hasVelo, &b_hasVelo);
  fChain->SetBranchAddress("isDown", &isDown, &b_isDown);
  fChain->SetBranchAddress("isDown_noVelo", &isDown_noVelo, &b_isDown_noVelo);
  fChain->SetBranchAddress("isLong", &isLong, &b_isLong);
  fChain->SetBranchAddress("key", &key, &b_key);
  fChain->SetBranchAddress("isLong_andUT", &isLong_andUT, &b_isLong_andUT);
  fChain->SetBranchAddress("p", &p, &b_p);
  fChain->SetBranchAddress("pt", &pt, &b_pt);
  fChain->SetBranchAddress("pid", &pid, &b_pid);
  fChain->SetBranchAddress("eta", &eta, &b_eta);
  fChain->SetBranchAddress("ovtx_x", &ovtx_x, &b_ovtx_x);
  fChain->SetBranchAddress("ovtx_y", &ovtx_y, &b_ovtx_y);
  fChain->SetBranchAddress("ovtx_z", &ovtx_z, &b_ovtx_z);
  fChain->SetBranchAddress("fromBeautyDecay", &fromBeautyDecay, &b_fromBeautyDecay);
  fChain->SetBranchAddress("fromCharmDecay", &fromCharmDecay, &b_fromCharmDecay);
  fChain->SetBranchAddress("fromStrangeDecay", &fromStrangeDecay, &b_fromStrangeDecay);
  fChain->SetBranchAddress("DecayOriginMother_pid", &DecayOriginMother_pid, &b_DecayOriginMother_pid);

  
  fChain->SetBranchAddress("UT_lhcbID", &UT_lhcbID, &b_UT_lhcbID);
  fChain->SetBranchAddress("FT_lhcbID", &FT_lhcbID, &b_FT_lhcbID);
  fChain->SetBranchAddress("Velo_lhcbID", &Velo_lhcbID, &b_Velo_lhcbID);
   
  Velo_lhcbID = 0;
  FT_lhcbID = 0;
  UT_lhcbID = 0;

  // Loop over tree containing MCPs of one event
  Long64_t maxEntries = fChain->GetTree()->GetEntries();
  for(Long64_t entry = 0; entry< maxEntries ; ++entry){
    fChain->GetTree()->GetEntry(entry);
    if( p<0) continue;  // Hits not associated to an MCP are stored with p < 0
    //Velo
    if ( trackType == "Velo" && !hasVelo ) continue;
    if ( trackType == "VeloUT" && !(hasVelo && hasUT) ) continue;
        
    VelopixEvent::MCP mcp;
    mcp.key = key;
    mcp.id = pid;
    mcp.p = p;
    mcp.pt = pt;
    mcp.eta = eta;
    //mcp.phi = phi; // not yet available in Ntuple
    mcp.islong = isLong;
    mcp.isdown = isDown;
    mcp.isvelo = hasVelo;
    mcp.isut = hasUT;
    mcp.strangelong = fromStrangeDecay && isLong;
    mcp.strangedown = fromStrangeDecay && isDown;
    mcp.fromb = fromBeautyDecay;
    mcp.fromd = fromCharmDecay;
    
    std::vector<uint32_t> hits;
    for(int index = 0; index < Velo_lhcbID->size(); index++) {
      hits.push_back( Velo_lhcbID->at(index) );
    }
    mcp.numHits = (uint32_t)hits.size();
    mcp.hits = hits;
    
    event.mcps.push_back( mcp );

  } // loop over MCPs
  
}
					 

std::vector<VelopixEvent> VelopixEventReader::readFolder (
  const std::string& foldername,
  const bool& fromNtuple,
  const std::string& trackType,
  uint nFiles,
  const bool checkEvents
  ) {

  std::vector< std::string > folderContents = getFolderContents( foldername, fromNtuple, nFiles );

  uint requestedFiles = nFiles==0 ? folderContents.size() : nFiles;

  info_cout << "Requested " << requestedFiles << " files" << std::endl;

  if ( requestedFiles > folderContents.size() ) {
    error_cout << "ERROR: requested " << requestedFiles << " files, but only " << folderContents.size() << " files are present" << std::endl;
    exit(-1);
  }
   
  std::vector<VelopixEvent> input;
  int readFiles = 0;
  for (uint i=0; i<requestedFiles; ++i) {
    // Read event #i in the list and add it to the inputs
    // if more files are requested than present in folder, read them again
    std::string readingFile = folderContents[i % folderContents.size()];
    debug_cout << "Reading MC event " << readingFile << std::endl;
    
    VelopixEvent event;
    if ( !fromNtuple ) {
      std::vector<uint8_t> inputContents;
      readFileIntoVector(foldername + "/" + readingFile, inputContents);
      event = VelopixEvent(inputContents, false);
    }
    else if ( fromNtuple )
      readNtupleIntoVelopixEvent(foldername + "/" + readingFile, trackType, event);
      
    if ( i == 0 )
      event.print();
       
    input.emplace_back(event);
        
    readFiles++;
    if ((readFiles % 100) == 0) {
      std::cout << "." << std::flush;
    }

  }

  info_cout << std::endl << input.size() << " files read" << std::endl << std::endl;
  return input;
}

		     
