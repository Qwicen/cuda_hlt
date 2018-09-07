/** @file MCAssociator.h
 *
 * @brief a simple MC associator
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-18
 */

#pragma once

#include <map>
#include <vector>
#include <utility>
#include <algorithm>

#include "LHCbID.h"
#include "MCParticle.h"

#include "Logger.h"


/// simple MC associator
class MCAssociator
{
    private:
        using LHCbIDWithIndex = std::pair<LHCbID, uint>;
        using AssocMap = std::vector<LHCbIDWithIndex>;

        /// internal structure with index into particles and weight
        struct MCParticleWithWeight {
            std::size_t m_idx;
            float m_w;
            MCParticleWithWeight(std::size_t idx, float w) : m_idx(idx), m_w(w)
            {}
            MCParticleWithWeight(const MCParticleWithWeight&) = default;
            MCParticleWithWeight(MCParticleWithWeight&&) = default;
            MCParticleWithWeight& operator=(const MCParticleWithWeight&) = default;
            MCParticleWithWeight& operator=(MCParticleWithWeight&&) = default;
        };
        const MCParticles& m_mcps; // keep a reference to MCParticles
        AssocMap m_map; // association LHCbID -> MCParticle index

        // little helper which does the hard work
        AssocMap::const_iterator find(LHCbID id) const noexcept;

    public:
        MCAssociator(const MCParticles& mcps);
        MCAssociator(const MCAssociator&) = default;
        MCAssociator(MCAssociator&&) = default;
        MCAssociator& operator=(const MCAssociator&) = default;
        MCAssociator& operator=(MCAssociator&&) = default;

        /// result of an MC association
        class MCAssocResult {
            private:
                using AssocVector = std::vector<MCParticleWithWeight>;
                AssocVector m_assoc;
                const MCParticles& m_mcps;

            public:
                MCAssocResult(AssocVector&& assocs, const MCParticles& mcps) :
                    m_assoc(assocs), m_mcps(mcps)
                {}
                MCAssocResult(const MCAssocResult&) = default;
                MCAssocResult(MCAssocResult&&) = default;
                MCAssocResult& operator=(const MCAssocResult&) = default;
                MCAssocResult& operator=(MCAssocResult&&) = default;

                bool empty() const noexcept
                { return m_assoc.empty(); }
                operator bool() const noexcept
                { return !empty(); }
                std::size_t size() const noexcept
                { return m_assoc.size(); }

                /// something to iterate over a MCAssocResult
                class iterator : private AssocVector::const_iterator
                {
                    private:
                        const MCParticles& m_mcps;
                    public:
                        iterator(AssocVector::const_iterator it,
                                const MCParticles& mcps) :
                            AssocVector::const_iterator(it), m_mcps(mcps)
                        {}
                        iterator(const iterator&) = default;
                        iterator(iterator&&) = default;
                        iterator& operator=(const iterator&) = default;
                        iterator& operator=(iterator&&) = default;
                        iterator& operator++() noexcept
                        {
                            ++reinterpret_cast<AssocVector::const_iterator&>(*this);
                            return *this;
                        }
                        iterator operator++(int) noexcept
                        {
                            auto retVal(*this);
                            ++reinterpret_cast<AssocVector::const_iterator&>(*this);
                            return retVal;
                        }
                        iterator& operator--() noexcept
                        {
                            --reinterpret_cast<AssocVector::const_iterator&>(*this);
                            return *this;
                        }
                        iterator operator--(int) noexcept
                        {
                            auto retVal(*this);
                            --reinterpret_cast<AssocVector::const_iterator&>(*this);
                            return retVal;
                        }
                        bool operator==(const iterator& other) const noexcept
                        {
                            const auto& a =
                                reinterpret_cast<const AssocVector::const_iterator&>(*this);
                            const auto& b =
                                reinterpret_cast<const AssocVector::const_iterator&>(other);
                            return a == b;
                        }
                        bool operator!=(const iterator& other) const noexcept
                        {
                            const auto& a =
                                reinterpret_cast<const AssocVector::const_iterator&>(*this);
                            const auto& b =
                                reinterpret_cast<const AssocVector::const_iterator&>(other);
                            return a != b;
                        }
                        bool operator<(const iterator& other) const noexcept
                        {
                            const auto& a =
                                reinterpret_cast<const AssocVector::const_iterator&>(*this);
                            const auto& b =
                                reinterpret_cast<const AssocVector::const_iterator&>(other);
                            return a < b;
                        }
                        bool operator<=(const iterator& other) const noexcept
                        {
                            const auto& a =
                                reinterpret_cast<const AssocVector::const_iterator&>(*this);
                            const auto& b =
                                reinterpret_cast<const AssocVector::const_iterator&>(other);
                            return a <= b;
                        }
                        bool operator>(const iterator& other) const noexcept
                        {
                            const auto& a =
                                reinterpret_cast<const AssocVector::const_iterator&>(*this);
                            const auto& b =
                                reinterpret_cast<const AssocVector::const_iterator&>(other);
                            return a > b;
                        }
                        bool operator>=(const iterator& other) const noexcept
                        {
                            const auto& a =
                                reinterpret_cast<const AssocVector::const_iterator&>(*this);
                            const auto& b =
                                reinterpret_cast<const AssocVector::const_iterator&>(other);
                            return a >= b;
                        }
                        std::pair<MCParticles::const_reference, float>
                        operator*() const noexcept
                        {
                            AssocVector::const_reference ref =
                                reinterpret_cast<const
                                AssocVector::const_iterator&>(
                                        *this).operator*();
                            return { m_mcps[ref.m_idx], ref.m_w };
                        }
                };
                iterator begin() const noexcept
                { return iterator(m_assoc.begin(), m_mcps); }
                iterator end() const noexcept
                { return iterator(m_assoc.end(), m_mcps); }
                std::pair<MCParticles::const_reference, float> front() const noexcept
                { return *begin(); }
                std::pair<MCParticles::const_reference, float> back() const noexcept
                { return *--end(); }
        };

    private:
        using AssocPreResult = std::map<std::size_t, std::size_t>;
        /// little helper for the final step of multi-MCP association
        MCAssocResult buildResult(const AssocPreResult& assocmap,
                std::size_t total) const noexcept;
    public:
        /// associate a single LHCbID
        MCAssocResult operator()(LHCbID id) const noexcept
        {
            auto it = find(id);
            if (m_map.end() == it) return MCAssocResult({}, m_mcps);
            return MCAssocResult({ {it->second , 1.f } }, m_mcps);
        }
        /// associate a range of LHCbIDs
        template <typename IT>
        MCAssocResult operator()(IT first, IT last, std::size_t &n_matched_total) const noexcept
        {
            AssocPreResult assoc;
            std::size_t total = 0;
            // count how often each particle appears
	    // and how many hits of the reconstructed track are matched
	    // to the MCP
            for (; last != first; ++first) {
	      auto it = find(*first);
                if (m_map.end() == it) continue;
                //std::cout << "Matched LHCbID to MCP: " << std::hex << *first << std::endl;
		++n_matched_total;
                ++assoc[it->second];
	    }

	    // bring the map into a more compact format
            return buildResult(assoc, n_matched_total);
        }
};
