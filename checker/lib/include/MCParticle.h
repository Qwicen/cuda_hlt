/** @file MCParticle.h
 *
 * @brief a simple MCParticle
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-18
 */

#pragma once

// Monte Carlo information
struct MCParticle {
  uint32_t key;
  int pid;
  float p;
  float pt;
  float eta;
  float phi;
  bool isLong;
  bool isDown;
  bool hasVelo;
  bool hasUT;
  bool hasSciFi;
  bool fromBeautyDecay;
  bool fromCharmDecay;
  bool fromStrangeDecay;
  uint32_t numHits;
  std::vector<uint32_t> hits;

};

using MCParticles = std::vector<MCParticle>;
