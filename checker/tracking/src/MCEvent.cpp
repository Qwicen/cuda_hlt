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

#include "MCEvent.h"

MCEvent::MCEvent(const std::vector<char>& event, const bool checkEvent)
{
  uint8_t* input = (uint8_t*) event.data();

  uint32_t number_mcp = *((uint32_t*) input);
  input += sizeof(uint32_t);
  // debug_cout << "num MCPs = " << number_mcp << std::endl;
  for (uint32_t i = 0; i < number_mcp; ++i) {
    MCParticle p;
    p.key = *((uint32_t*) input);
    input += sizeof(uint32_t);
    p.pid = *((uint32_t*) input);
    input += sizeof(uint32_t);
    p.p = *((float*) input);
    input += sizeof(float);
    p.pt = *((float*) input);
    input += sizeof(float);
    p.eta = *((float*) input);
    input += sizeof(float);
    p.phi = *((float*) input);
    input += sizeof(float);
    p.isLong = (bool) *((int8_t*) input);
    input += sizeof(int8_t);
    p.isDown = (bool) *((int8_t*) input);
    input += sizeof(int8_t);
    p.hasVelo = (bool) *((int8_t*) input);
    input += sizeof(int8_t);
    p.hasUT = (bool) *((int8_t*) input);
    input += sizeof(int8_t);
    p.hasSciFi = (bool) *((int8_t*) input);
    input += sizeof(int8_t);
    p.fromBeautyDecay = (bool) *((int8_t*) input);
    input += sizeof(int8_t);
    p.fromCharmDecay = (bool) *((int8_t*) input);
    input += sizeof(int8_t);
    p.fromStrangeDecay = (bool) *((int8_t*) input);
    input += sizeof(int8_t);
    p.nPV = *((uint32_t*) input);
    input += sizeof(uint32_t);

    int num_Velo_hits = *((uint32_t*) input);
    input += sizeof(uint32_t);
    std::vector<uint32_t> hits;
    std::copy_n((uint32_t*) input, num_Velo_hits, std::back_inserter(hits));
    input += sizeof(uint32_t) * num_Velo_hits;

    // Add the mcp to velo_mcps
    p.numHits = (uint) hits.size();
    p.hits = hits;
    if (num_Velo_hits > 0) {
      velo_mcps.push_back(p);
    }

    int num_UT_hits = *((uint32_t*) input);
    input += sizeof(uint32_t);
    std::copy_n((uint32_t*) input, num_UT_hits, std::back_inserter(hits));
    input += sizeof(uint32_t) * num_UT_hits;

    // Add the mcp to ut_mcps
    p.numHits = (uint) hits.size();
    p.hits = hits;
    if (num_Velo_hits > 0 || num_UT_hits > 0) {
      ut_mcps.push_back(p);
    }

    int num_SciFi_hits = *((uint32_t*) input);
    input += sizeof(uint32_t);
    std::copy_n((uint32_t*) input, num_SciFi_hits, std::back_inserter(hits));
    input += sizeof(uint32_t) * num_SciFi_hits;

    // Add the mcp to scifi_mcps
    p.numHits = (uint) hits.size();
    p.hits = hits;
    if (num_Velo_hits > 0 || num_UT_hits > 0 || num_SciFi_hits > 0) {
      scifi_mcps.push_back(p);
    }
  }

  size = input - (uint8_t*) event.data();

  if (size != event.size()) {
    throw StrException(
      "Size mismatch in event deserialization: " + std::to_string(size) + " vs " + std::to_string(event.size()));
  }

  if (checkEvent) {
    // Check all floats are valid
    const auto check_mcp = [](const MCParticle& mcp) {
      assert(!std::isnan(mcp.p));
      assert(!std::isnan(mcp.pt));
      assert(!std::isnan(mcp.eta));
      assert(!std::isinf(mcp.p));
      assert(!std::isinf(mcp.pt));
      assert(!std::isinf(mcp.eta));
    };

    for (const auto& mcp : velo_mcps) {
      check_mcp(mcp);
    }
    for (const auto& mcp : ut_mcps) {
      check_mcp(mcp);
    }
    for (const auto& mcp : scifi_mcps) {
      check_mcp(mcp);
    }
  }
}

void MCEvent::print(const MCParticles& mcps) const
{
  info_cout << " #MC particles " << mcps.size() << std::endl;

  // Print first MCParticle
  if (mcps.size() > 0) {
    auto& p = mcps[0];
    info_cout << " First MC particle" << std::endl
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

template<>
const MCParticles& MCEvent::mc_particles<TrackCheckerVelo>() const
{
  return velo_mcps;
}

template<>
const MCParticles& MCEvent::mc_particles<TrackCheckerVeloUT>() const
{
  return ut_mcps;
}

template<>
const MCParticles& MCEvent::mc_particles<TrackCheckerForward>() const
{
  return scifi_mcps;
}