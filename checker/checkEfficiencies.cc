/** @file velo-phi-drdz2.cc
 *
 * @brief unit tests for phi-(drdz)^2 tracking for the upgrade velo
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-06
 */

#include <cassert>
#include <iostream>
#include <fstream>

#include "velopix-input-reader.h"
#include "track_input_reader.h"
#include "MCAssociator.h"
#include "TrackChecker.h"

int main()
{
  /* Tracks to be checked */
  std::ifstream tracks_in ("tracks_checker_out.txt" );
  std::vector< trackChecker::Tracks > all_tracks = read_input_tracks( tracks_in );
  
  /* MC information */
  std::vector<VelopixEvent> events = VelopixEventReader::readFolder("../input_checker", 20, true );
  
  TrackChecker trackChecker;
  uint64_t evnum = 1;
  for (const auto& ev: events) {
    std::cout << "Event " << evnum << std::endl;
    auto pixels = ev.soaHits();
    auto mcps = ev.mcparticles();
    MCAssociator mcassoc(mcps);

    trackChecker::Tracks tracks = all_tracks[evnum-1];
    std::cout << "INFO: found " << tracks.size() << " reconstructed tracks" <<
     " and " << mcps.size() << " MC particles " << std::endl;

    trackChecker(tracks, mcassoc, mcps);
        
    ++evnum;
  }
  return 0;
}

// vim: sw=4:tw=78:ft=cpp:et
