/** @file MCAssociator.cpp
 *
 * @brief a simple MC associator
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-18
 */

#include <numeric>

#include "MCAssociator.h"

MCAssociator::MCAssociator(const MCParticles& mcps) :
  m_mcps(mcps)
{
  // work out how much space we need
  const std::size_t total = std::accumulate(mcps.begin(), mcps.end(), 0,
      [] (std::size_t acc, MCParticles::const_reference mcp) noexcept
      { return acc + mcp.numHits; });
  m_map.reserve(total);
  // build association LHCbID -> MCParticle index
  std::size_t idx = 0;
  for (auto mcp: mcps) {
    for (auto id: mcp.hits) {
      m_map.emplace_back(id, idx);
    }
    ++idx;
  }
  // sort map by LHCbID for fast lookups
  std::sort(m_map.begin(), m_map.end(),
      [] (const LHCbIDWithIndex& a, const LHCbIDWithIndex& b) noexcept
      { return a.first < b.first; });
}

MCAssociator::AssocMap::const_iterator MCAssociator::find(LHCbID id) const noexcept
{
  // dcampora: Relaxed check
  // auto it = std::find_if(m_map.begin(), m_map.end(), [&id] (const LHCbIDWithIndex& a) {
  //   return a.first == id;
  // });
  // return it;

  auto it = std::lower_bound(m_map.begin(), m_map.end(), id,
      [] (const LHCbIDWithIndex& a, const LHCbID& b) noexcept
      { return a.first < b; });
  if (m_map.end() == it) return it;
  if (id != it->first) return m_map.end();
  return it;
}

MCAssociator::MCAssocResult MCAssociator::buildResult(
    const MCAssociator::AssocPreResult& assocmap,
    std::size_t total) const noexcept
{
  std::vector<MCParticleWithWeight> retVal;
  retVal.reserve(assocmap.size());
  for (auto&& el: assocmap)
    retVal.emplace_back(
        el.first, float(el.second) / float(total));
  // sort such that high weights come first
  std::sort(retVal.begin(), retVal.end(),
      [] (const MCParticleWithWeight& a,
        const MCParticleWithWeight& b) noexcept
      { return a.m_w > b.m_w; });
  return MCAssocResult{std::move(retVal), m_mcps};
}
