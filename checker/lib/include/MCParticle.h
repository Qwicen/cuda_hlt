/** @file MCParticle.h
 *
 * @brief SOA MC Particle
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-18
 */

#pragma once

#include <cstdint>
#include <array>

#include "boost/range/iterator_range.hpp"

#include "../SOAContainer/include/SOAContainer.h"
#include "LHCbID.h"

namespace MCParticleDesc {
    SOAFIELD_TRIVIAL(p, p, float);
    SOAFIELD_TRIVIAL(pt, pt, float);
    SOAFIELD_TRIVIAL(eta, eta, float);
    SOAFIELD_TRIVIAL(phi, phi, float);
    SOAFIELD_TRIVIAL(pid, pid, int32_t);
    SOAFIELD_TRIVIAL(key, key, uint32_t);
    SOAFIELD_TRIVIAL(nIDs, nIDs, uint32_t);
    SOAFIELD_TRIVIAL(allids, allids, SomeLHCbIDs);
    enum Flags {
        Long = 0x01, Down = 0x02,
        Velo = 0x04, UT = 0x08,
        StrangeLong = 0x10, StrangeDown = 0x20,
        FromB = 0x40, FromD = 0x80
    };
    SOAFIELD(flags, uint32_t,
            SOAFIELD_ACCESSORS(flags);
            bool isLong() const noexcept
            { return this->flags() & Long; }
            bool isDown() const noexcept
            { return this->flags() & Down; }
            bool isVelo() const noexcept
            { return this->flags() & Velo; }
            bool isUT() const noexcept
            { return this->flags() & UT; }
            bool isStrangeLong() const noexcept
            { return this->flags() & StrangeLong; }
            bool isStrangeDown() const noexcept
            { return this->flags() & StrangeDown; }
            bool isFromB() const noexcept
            { return this->flags() & FromB; }
            bool isFromD() const noexcept
            { return this->flags() & FromD; }
            );

    SOASKIN(Skin, p, pt, eta, phi, pid, key, nIDs, allids, flags) {
        SOASKIN_INHERIT_DEFAULT_METHODS(Skin);
                ConstSomeLHCbIDRange ids() const noexcept
        {
            return boost::make_iterator_range(
                    this->allids().begin(),
                    this->allids().begin() + this->nIDs());
        }
        SomeLHCbIDRange ids() noexcept
        {
            return boost::make_iterator_range(
                    this->allids().begin(),
                    this->allids().begin() + this->nIDs());
        }
    };
}

using MCParticles = SOA::Container<std::vector, MCParticleDesc::Skin>;
using MCParticleRange = boost::iterator_range<MCParticles::iterator>;
