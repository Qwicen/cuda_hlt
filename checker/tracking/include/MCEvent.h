/** @file MCEvent.h
 *
 * @brief a reader of MC events
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-18
 *
 * 2018-07 Dorothea vom Bruch: updated to run over different track types,
 * take input from Renato Quagliani's TrackerDumper
 */

#pragma once

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include "Common.h"
#include "Logger.h"
#include "MCParticle.h"
#include "TrackChecker.h"

struct MCEvent {
  MCParticles velo_mcps;
  MCParticles ut_mcps;
  MCParticles scifi_mcps;

  uint32_t size;

  // Constructor
  MCEvent() {};
  MCEvent(const std::vector<char>& _event, const bool checkFile = true);

  /**
   * @brief Print a specific set of MC particles.
   */
  void print(const MCParticles& mcps) const;

  template<typename T>
  const MCParticles& mc_particles() const;

  /**
   * @brief Print the set of MC particles of type T.
   */
  template<typename T>
  void print() const
  {
    print(mc_particles<T>());
  }
};

using MCEvents = std::vector<MCEvent>;
