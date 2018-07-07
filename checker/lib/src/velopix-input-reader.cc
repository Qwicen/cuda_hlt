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

#include "velopix-input-reader.h"

VelopixEvent::VelopixEvent(const std::vector<char>& event, const bool checkEvent) {
  uint8_t* input = (uint8_t*) event.data();

  // Event
  numberOfModules  = *((uint32_t*)input); input += sizeof(uint32_t);
  numberOfHits   = *((uint32_t*)input); input += sizeof(uint32_t);
  std::copy_n((float*) input, numberOfModules, std::back_inserter(module_Zs)); input += sizeof(float) * numberOfModules;
  std::copy_n((uint32_t*) input, numberOfModules, std::back_inserter(module_hitStarts)); input += sizeof(uint32_t) * numberOfModules;
  std::copy_n((uint32_t*) input, numberOfModules, std::back_inserter(module_hitNums)); input += sizeof(uint32_t) * numberOfModules;
  std::copy_n((uint32_t*) input, numberOfHits, std::back_inserter(hit_IDs)); input += sizeof(uint32_t) * numberOfHits;
  std::copy_n((float*) input, numberOfHits, std::back_inserter(hit_Xs)); input += sizeof(float) * numberOfHits;
  std::copy_n((float*) input, numberOfHits, std::back_inserter(hit_Ys)); input += sizeof(float) * numberOfHits;
  std::copy_n((float*) input, numberOfHits, std::back_inserter(hit_Zs)); input += sizeof(float) * numberOfHits;

  // Monte Carlo
  uint32_t number_mcp = *((uint32_t*)  input); input += sizeof(uint32_t);
  for (uint32_t i=0; i<number_mcp; ++i) {
    MCParticle p;
    p.m_key      = *((uint32_t*)  input); input += sizeof(uint32_t);
    p.m_id       = *((uint32_t*)  input); input += sizeof(uint32_t);
    p.m_p      = *((float*)  input); input += sizeof(float);
    p.m_pt       = *((float*)  input); input += sizeof(float);
    p.m_eta      = *((float*)  input); input += sizeof(float);
    p.m_phi      = *((float*)  input); input += sizeof(float);
    p.m_islong     = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.m_isdown     = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.m_isvelo     = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.m_isut     = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.m_strangelong  = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.m_strangedown  = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.m_fromb    = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.m_fromd    = (bool) *((int8_t*)  input); input += sizeof(int8_t);
    p.m_numHits    = *((uint32_t*)  input); input += sizeof(uint32_t);
    std::copy_n((uint32_t*) input, p.nIDs(), std::back_inserter(p.m_hits)); input += sizeof(uint32_t) * p.nIDs();
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
      assert(!std::isnan(mcp.p()));
      assert(!std::isnan(mcp.pt()));
      assert(!std::isnan(mcp.eta()));
      assert(!std::isnan(mcp.phi()));
      assert(!std::isinf(mcp.p()));
      assert(!std::isinf(mcp.pt()));
      assert(!std::isinf(mcp.eta()));
      assert(!std::isinf(mcp.phi()));
      // Check all IDs in MC particles exist in hit_IDs
      for (size_t i=0; i<mcp.nIDs(); ++i) {
        auto hit = mcp.m_hits[i];
        if (std::find(hit_IDs.begin(), hit_IDs.end(), hit) == hit_IDs.end()) {
          throw StrException("The following MC particle hit ID was not found in hit_IDs: " + std::to_string(hit));
        }
      }
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

  // Print first MCParticle
  if (mcps.size() > 0) {
    auto& p = mcps[0];
    std::cout << " First MC particle" << std::endl
      << "  key " << p.key() << std::endl
      << "  id " << p.pid() << std::endl
      << "  p " << p.p() << std::endl
      << "  pt " << p.pt() << std::endl
      << "  eta " << p.eta() << std::endl
      << "  phi " << p.phi() << std::endl
      << "  islong " << p.isLong() << std::endl
      << "  isdown " << p.isDown() << std::endl
      << "  isvelo " << p.isVelo() << std::endl
      << "  isut " << p.isUT() << std::endl
      << "  strangelong " << p.isStrangeLong() << std::endl
      << "  strangedown " << p.isStrangeDown() << std::endl
      << "  fromb " << p.isFromB() << std::endl
      << "  fromd " << p.isFromD() << std::endl
      << "  numHits " << p.nIDs() << std::endl
      << "  hits " << strVector(p.m_hits, p.nIDs()) << std::endl;
  }
}

MCParticles VelopixEvent::mcparticles() const
{
  return mcps;
}
