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

VelopixEvent::VelopixEvent(const std::vector<char>& event, const std::string& trackType, const bool checkEvent) {
  uint8_t* input = (uint8_t*) event.data();

  uint32_t number_mcp = *((uint32_t*)  input); input += sizeof(uint32_t);
  //std::cout << "num MCPs = " << number_mcp << std::endl;
  for (uint32_t i=0; i<number_mcp; ++i) {
    MCParticle p;
    p.key               = *((uint32_t*)  input); input += sizeof(uint32_t);
    p.pid               = *((uint32_t*)  input); input += sizeof(uint32_t);
    p.p                 = *((float*)  input); input += sizeof(float);
    p.pt                = *((float*)  input); input += sizeof(float);
    p.eta               = *((float*)  input); input += sizeof(float);
    //p.phi             = *((float*)  input); input += sizeof(float); // phi not available now
    p.isLong            = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.isDown            = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.hasVelo           = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.hasUT             = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.hasSciFi          = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.fromBeautyDecay   = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.fromCharmDecay    = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.fromStrangeDecay  = (bool) *((int8_t*)  input); input += sizeof(int8_t);

   
    int num_Velo_hits = *((uint32_t*)  input); input += sizeof(uint32_t);
    std::vector<uint32_t> velo_hits;
    std::copy_n((uint32_t*) input, num_Velo_hits, std::back_inserter(velo_hits));
    input += sizeof(uint32_t) * num_Velo_hits;
    int num_UT_hits = *((uint32_t*)  input); input += sizeof(uint32_t);
    std::vector<uint32_t> UT_hits;
    std::copy_n((uint32_t*) input, num_UT_hits, std::back_inserter(UT_hits));
    input += sizeof(uint32_t) * num_UT_hits;
    int num_SciFi_hits = *((uint32_t*)  input); input += sizeof(uint32_t);
    std::vector<uint32_t> SciFi_hits;
    std::copy_n((uint32_t*) input, num_SciFi_hits, std::back_inserter(SciFi_hits));
    input += sizeof(uint32_t) * num_SciFi_hits;

    /* Only save the hits relevant for the track type we are checking
       -> get denominator of efficiency right
     */
    std::vector<uint32_t> hits;
    if ( trackType == "Velo" || trackType == "VeloUT" || trackType == "SciFi" )
      for(int index = 0; index < velo_hits.size(); index++) {
	hits.push_back( velo_hits.at(index) );
      }
    
    if ( trackType == "VeloUT" || trackType == "SciFi" )
      for(int index = 0; index < UT_hits.size(); index++) {
	hits.push_back( UT_hits.at(index) );
      }
    
    if ( trackType == "SciFi" )
      for(int index = 0; index < SciFi_hits.size(); index++) {
	hits.push_back( SciFi_hits.at(index) );
      }
    
    p.numHits    = int( hits.size() );
    p.hits = hits;
    mcps.push_back(p);
  }

  size = input - (uint8_t*) event.data();

  if (size != event.size()) {
    throw StrException("Size mismatch in event deserialization: " +
        std::to_string(size) + " vs " + std::to_string(event.size()));
  }

    if (checkEvent) {
      // Check all floats are valid
      for (auto& mcp : mcps) {
        assert(!std::isnan(mcp.p ));
        assert(!std::isnan(mcp.pt));
        assert(!std::isnan(mcp.eta));
        assert(!std::isnan(mcp.phi));
        assert(!std::isinf(mcp.p));
        assert(!std::isinf(mcp.pt));
        assert(!std::isinf(mcp.eta));
        assert(!std::isinf(mcp.phi));
      }
    }
     
}

void VelopixEvent::print() const {
  std::cout << " #MC particles " << mcps.size() << std::endl;

  // Print first MCParticle
  if (mcps.size() > 0) {
    auto& p = mcps[0];
    std::cout << " First MC particle" << std::endl
      << "  key " << p.key << std::endl
      << "  id " << p.pid << std::endl
      << "  p " << p.p << std::endl
      << "  pt " << p.pt << std::endl
      << "  eta " << p.eta << std::endl
      << "  phi " << p.phi << std::endl
      << "  isLong " << p.isLong << std::endl
      << "  isDown " << p.isDown << std::endl
      << "  hasVelo " << p.hasVelo << std::endl
      << "  hasUT " << p.hasUT << std::endl
      << "  hasSciFi " << p.hasSciFi << std::endl
      << "  fromBeautyDecay " << p.fromBeautyDecay << std::endl
      << "  fromCharmDecay " << p.fromCharmDecay << std::endl
      << "  fromStrangeDecay " << p.fromStrangeDecay << std::endl
      << "  numHits " << p.numHits << std::endl
      << "  hits " << strVector(p.hits, p.numHits) << std::endl;
  }
}

MCParticles VelopixEvent::mcparticles() const
{
  return mcps;
}


/* Use Ntuple created by PrTrackerDumper tool, 
   written by Renato Quagliani,
   Use code from https://gitlab.cern.ch/rquaglia/PrOfflineStudies
   by Renato Quagliani to read the input
*/
void readNtupleIntoVelopixEvent(
  const std::string& filename,
  const std::string& trackType,
  VelopixEvent& event							    
 ) {

  // Check if file exists
  if (!exists_test(filename)){
    throw StrException("Error: File " + filename + " does not exist.");
  }
   
  TFile *file = new TFile(filename.data(),"READ");
  TTree *tree = (TTree*)file->Get("Hits_detectors");
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

    // to do: do we really need this check?
    if ( trackType == "Velo" && !hasVelo ) continue;
    if ( trackType == "VeloUT" && !(hasVelo && hasUT) ) continue;
        
    MCParticle mcp;
    mcp.key = key;
    mcp.pid = pid;
    mcp.p = p;
    mcp.pt = pt;
    mcp.eta = eta;
    //mcp.phi = phi; // not yet available in Ntuple
    mcp.isLong = isLong;
    mcp.isDown = isDown;
    mcp.hasVelo = hasVelo;
    mcp.hasUT = hasUT;
    mcp.fromBeautyDecay = fromBeautyDecay;
    mcp.fromCharmDecay = fromCharmDecay;
    mcp.fromStrangeDecay = fromStrangeDecay;
    
    std::vector<uint32_t> hits;
    if ( trackType == "Velo" || trackType == "VeloUT" || trackType == "SciFi" )
      for(int index = 0; index < Velo_lhcbID->size(); index++) {
	hits.push_back( Velo_lhcbID->at(index) );
      }

    if ( trackType == "VeloUT" || trackType == "SciFi" )
      for(int index = 0; index < UT_lhcbID->size(); index++) {
	hits.push_back( UT_lhcbID->at(index) );
      }
    
    if ( trackType == "SciFi" )
      for(int index = 0; index < FT_lhcbID->size(); index++) {
	hits.push_back( FT_lhcbID->at(index) );
      }
        
    mcp.numHits = (uint32_t)hits.size();
    mcp.hits = hits;
    
    event.mcps.push_back( mcp );

  } // loop over MCPs

  file->Close();
  delete file;
}
 

std::vector<VelopixEvent> read_mc_folder (
  const std::string& foldername,
  const bool& fromNtuple,
  const std::string& trackType,
  uint number_of_files,
  const uint start_event_offset,
  const bool checkEvents
) {
  std::vector<std::string> folderContents = list_folder(foldername, fromNtuple);
  
  uint requestedFiles = number_of_files==0 ? folderContents.size() : number_of_files;
  verbose_cout << "Requested " << requestedFiles << " files" << std::endl;

  if ( requestedFiles > folderContents.size() ) {
    error_cout << "ERROR: requested " << requestedFiles << " files, but only " << folderContents.size() << " files are present" << std::endl;
    exit(-1);
  }
  
  std::vector<VelopixEvent> input;
  int readFiles = 0;
  for (uint i = start_event_offset; i < requestedFiles + start_event_offset; ++i) {
    // Read event #i in the list and add it to the inputs
    // if more files are requested than present in folder, read them again
    std::string readingFile = folderContents[i % folderContents.size()];
  
    VelopixEvent event;
    if ( !fromNtuple ) {
      std::vector<char> inputContents;
      readFileIntoVector(foldername + "/" + readingFile, inputContents);
      event = VelopixEvent(inputContents, trackType, checkEvents);
    }
    else if ( fromNtuple )
      readNtupleIntoVelopixEvent(foldername + "/" + readingFile, trackType, event);
      
    if ( i == 0 )
      event.print();
       
    input.emplace_back(event);

    readFiles++;
    if ((readFiles % 100) == 0) {
      info_cout << "." << std::flush;
    }
  }

  info_cout << std::endl << input.size() << " files read" << std::endl << std::endl;
  return input;
}
