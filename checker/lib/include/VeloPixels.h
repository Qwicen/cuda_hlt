/** @file VeloPixels.h
 *
 * @brief SOA Velo Pixels
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-06
 */

#pragma once

#include <cstdint>

#include "boost/range/iterator_range.hpp"

#include "../SOAContainer/include/SOAContainer.h"
#include "LHCbID.h"

namespace VeloPixelDesc {
    SOAFIELD_TRIVIAL(x, x, float);
    SOAFIELD_TRIVIAL(y, y, float);
    SOAFIELD_TRIVIAL(z, z, float);
    SOAFIELD_TRIVIAL(id, id, LHCbID);
    SOASKIN_TRIVIAL(Skin, x, y, z, id);
}

using VeloPixels = SOA::Container<std::vector, VeloPixelDesc::Skin>;
using VeloPixelsRange = boost::iterator_range<VeloPixels::iterator>;

// vim: sw=4:tw=78:ft=cpp:et
