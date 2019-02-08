/** @file MCParticle.h
 *
 * @brief a simple MCParticle
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-18
 *
 * 07-2018: updated categories and names, Dorothea vom Bruch
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
  uint32_t nPV; // # of reconstructible primary vertices in event
  std::vector<uint32_t> hits;

  bool isElectron() const { return 11 == std::abs(pid); };
  bool inEta2_5() const { return (eta < 5. && eta > 2.); };
};

using MCParticles = std::vector<MCParticle>;
