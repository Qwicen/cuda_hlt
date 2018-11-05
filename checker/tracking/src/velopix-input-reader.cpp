/** @file velopix-input-reader.cpp
 *
 * @brief reader of velopix input files
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-08
 *
 * 2018-07 Dorothea vom Bruch: updated to run over different track types, 
 * take input from Renato Quagliani's TrackerDumper
 */

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <dirent.h>
#include "velopix-input-reader.h"

VelopixEvent::VelopixEvent(const std::vector<char>& event, const std::string& trackType, const bool checkEvent) {
  uint8_t* input = (uint8_t*) event.data();

  uint32_t number_mcp = *((uint32_t*)  input); input += sizeof(uint32_t);
  //debug_cout << "num MCPs = " << number_mcp << std::endl;
  for (uint32_t i=0; i<number_mcp; ++i) {
    MCParticle p;
    p.key               = *((uint32_t*)  input); input += sizeof(uint32_t);
    p.pid               = *((uint32_t*)  input); input += sizeof(uint32_t);
    p.p                 = *((float*)  input); input += sizeof(float);
    p.pt                = *((float*)  input); input += sizeof(float);
    p.eta               = *((float*)  input); input += sizeof(float);
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

    // if ( trackType == "Velo" && !p.hasVelo ) continue;
    // if ( trackType == "VeloUT" && !(p.hasVelo && p.hasUT) ) continue;
    // if ( trackType == "Forward" && !(p.hasVelo && p.hasUT && p.hasSciFi) ) continue;
   
    /* Only save the hits relevant for the track type we are checking
       -> get denominator of efficiency right
     */
    std::vector<uint32_t> hits;
    if ( trackType == "Velo" || trackType == "VeloUT" || trackType == "Forward" )
      for(int index = 0; index < velo_hits.size(); index++) {
	hits.push_back( velo_hits.at(index) );
      }
    
    if ( trackType == "VeloUT" || trackType == "Forward" )
      for(int index = 0; index < UT_hits.size(); index++) {
	hits.push_back( UT_hits.at(index) );
      }
    
    if ( trackType == "Forward" )
      for(int index = 0; index < SciFi_hits.size(); index++) {
	hits.push_back( SciFi_hits.at(index) );
      }
    
    p.numHits    = uint( hits.size() );
    p.hits = hits;
    if ( !(num_Velo_hits == 0 && num_UT_hits == 0 && num_SciFi_hits == 0) ) {
      //debug_cout << "MCP has " << num_Velo_hits << " Velo hits, " << num_UT_hits << " UT hits, " << num_SciFi_hits << " SciFi hits" << std::endl; 
      mcps.push_back(p);
    }
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
        assert(!std::isinf(mcp.p));
        assert(!std::isinf(mcp.pt));
        assert(!std::isinf(mcp.eta));
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
      << "  isLong " << p.isLong << std::endl
      << "  isDown " << p.isDown << std::endl
      << "  hasVelo " << p.hasVelo << std::endl
      << "  hasUT " << p.hasUT << std::endl
      << "  hasSciFi " << p.hasSciFi << std::endl
      << "  fromBeautyDecay " << p.fromBeautyDecay << std::endl
      << "  fromCharmDecay " << p.fromCharmDecay << std::endl
      << "  fromStrangeDecay " << p.fromStrangeDecay << std::endl
      << "  numHits " << p.numHits << std::endl;
      }
}

MCParticles VelopixEvent::mcparticles() const
{
  return mcps;
}

std::tuple<bool, std::vector<VelopixEvent>> read_mc_folder (
  const std::string& foldername,
  const std::string& trackType,
  uint number_of_files,
  const uint start_event_offset,
  const bool checkEvents
) {
  std::vector<std::string> folderContents = list_folder(foldername);
  
  uint requestedFiles = number_of_files==0 ? folderContents.size() : number_of_files;
  verbose_cout << "Requested " << requestedFiles << " files" << std::endl;

  if ( requestedFiles > folderContents.size() ) {
    error_cout << "Monte Carlo validation failed: Requested "
      << requestedFiles << " events, but only " << folderContents.size() << " Monte Carlo files are present."
      << std::endl << std::endl;

    return {false, {}};
  }
  
  std::vector<VelopixEvent> input;
  int readFiles = 0;
  for (uint i = start_event_offset; i < requestedFiles + start_event_offset; ++i) {
    // Read event #i in the list and add it to the inputs
    // if more files are requested than present in folder, read them again
    std::string readingFile = folderContents[i % folderContents.size()];
  
    VelopixEvent event;
    std::vector<char> inputContents;
    readFileIntoVector(foldername + "/" + readingFile, inputContents);
    event = VelopixEvent(inputContents, trackType, checkEvents);

    //debug_cout << "At VelopixEvent " << i << ": " << int(event.mcps.size()) << " MCPs" << std::endl;
    // if ( i == 0 && checkEvents )
    //      event.print();
       
    input.emplace_back(event);

    readFiles++;
    if ((readFiles % 100) == 0) {
      info_cout << "." << std::flush;
    }
  }

  info_cout << std::endl << input.size() << " files read" << std::endl << std::endl;
  return {true, input};
}
