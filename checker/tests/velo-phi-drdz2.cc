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

#include "velopix-input-reader.h"
#include "MCAssociator.h"
#include "TrackChecker.h"
#include "TrackSerializer.h"

int main()
{
    std::vector<VelopixEvent> events = VelopixEventReader::readFolder("../input");
    TrackSerializer trackWriter("../output/");
    TrackChecker trackChecker;
    //VeloPhiDrdz2 tracking;
    uint64_t evnum = 1;
    for (const auto& ev: events) {
        std::cout << "Event " << evnum << std::endl;
        auto pixels = ev.soaHits();
        auto mcps = ev.mcparticles();
        MCAssociator mcassoc(mcps);
        std::cout << "INFO: have " << pixels.size() << " pixels" << std::endl;
        //auto tracks = tracking(pixels, mcassoc);
        //std::cout << "INFO: found " << tracks.size() << " tracks" << std::endl;
        //trackChecker(tracks, mcassoc, mcps);
        //trackWriter(evnum, tracks);

        ++evnum;
    }
    return 0;
}

// vim: sw=4:tw=78:ft=cpp:et
