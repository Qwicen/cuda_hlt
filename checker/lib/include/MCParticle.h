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
  uint32_t m_key;
  uint32_t m_id;
  float m_p;
  float m_pt;
  float m_eta;
  float m_phi;
  bool m_islong;
  bool m_isdown;
  bool m_isvelo;
  bool m_isut;
  bool m_strangelong;
  bool m_strangedown;
  bool m_fromb;
  bool m_fromd;
  uint32_t m_numHits;
  std::vector<uint32_t> m_hits;

  uint32_t key() const noexcept
  { return m_key; }

  uint32_t pid() const noexcept
  { return m_id; }

  float p() const noexcept
  { return m_p; }

  float pt() const noexcept
  { return m_p; }

  float eta() const noexcept
  { return m_eta; }

  float phi() const noexcept
  { return m_phi; }

  bool isLong() const noexcept
  { return m_islong; }
  
  bool isDown() const noexcept
  { return m_isdown; }
  
  bool isVelo() const noexcept
  { return m_isvelo; }
  
  bool isUT() const noexcept
  { return m_isut; }
  
  bool isStrangeLong() const noexcept
  { return m_strangelong; }
  
  bool isStrangeDown() const noexcept
  { return m_strangedown; }
  
  bool isFromB() const noexcept
  { return m_fromb; }
  
  bool isFromD() const noexcept
  { return m_fromd; }

  size_t nIDs() const noexcept
  { return m_numHits; }
};

using MCParticles = std::vector<MCParticle>;
