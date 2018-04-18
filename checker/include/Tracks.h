/** @file Tracks.h
 *
 * @brief SOA Velo Tracks
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-06
 */

#pragma once

#include <array>
#include <cstdint>

#include "boost/range/iterator_range.hpp"

#include "SOAContainer.h"
#include "LHCbID.h"

namespace TracksDesc {
    SOAFIELD_TRIVIAL(x0, x0, float);
    SOAFIELD_TRIVIAL(y0, y0, float);
    SOAFIELD_TRIVIAL(z0, z0, float);
    SOAFIELD_TRIVIAL(tx, tx, float);
    SOAFIELD_TRIVIAL(ty, ty, float);
    SOAFIELD_TRIVIAL(nIDs, nIDs, uint32_t); // number of IDs
    SOAFIELD_TRIVIAL(allids, allids, SomeLHCbIDs);
    SOAFIELD(flags, uint32_t,
            SOAFIELD_ACCESSORS(flags);
            enum Flags{ Clone = 0x1 };
            bool isClone() const noexcept { return this->flags() & Clone; }
            void setClone(bool isClone = true) noexcept
            { this->flags() &= ~Clone; this->flags() |= (-uint32_t(isClone)) & Clone; }
    );
    SOASKIN(Skin, x0, y0, z0, tx, ty, nIDs, allids, flags) {
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
        void addIds(ConstSomeLHCbIDRange ids) noexcept
        {
            auto it = ids.begin();
            auto i = this->nIDs();
            for (; this->allids().size() != i; ++i) {
                if (ids.end() == it) break;
                *(this->allids().begin() + i) = *it++;
            }
            this->nIDs() = i;
        }

        float x(float z) const noexcept
        { return this->x0() + this->tx() * (z - this->z0()); }
        float y(float z) const noexcept
        { return this->y0() + this->ty() * (z - this->z0()); }
    };
}

using Tracks = SOA::Container<std::vector, TracksDesc::Skin>;
using TracksRange = boost::iterator_range<Tracks::iterator>;

// vim: sw=4:tw=78:ft=cpp:et
