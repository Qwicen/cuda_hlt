/** @file TrackSerializer.cc
 *
 * @brief simple binary track serializer
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-06
 */

#include <vector>
#include <cstdint>
#include <cassert>
#include <fstream>

#include "TrackSerializer.h"
#include "Tracks.h"

void TrackSerializer::operator()(uint64_t evnum, const Tracks& tracks) const
{
    uint32_t ntracks = tracks.size();
    uint32_t nIDs = 0;
    for (auto tr: tracks)
        nIDs += tr.nIDs();
    std::size_t totsz = sizeof(uint32_t) * (1 + ntracks + nIDs);
    std::vector<uint8_t> data(totsz, 0);
    uint32_t* p = reinterpret_cast<uint32_t*>(data.data());
    uint32_t* pend = reinterpret_cast<uint32_t*>(data.data() + totsz);
    *p++ = ntracks;
    assert(p <= pend);
    for (auto tr: tracks) {
        *p++ = tr.nIDs();
        assert(p <= pend);
        const auto ids = boost::make_iterator_range(
                tr.ids().begin(), tr.ids().begin() + tr.nIDs());
        for (auto id: ids) {
            *p++ = id;
            assert(p <= pend);
        }
    }
    std::string filename(m_pfx);
    {
        char buf[64];
        std::snprintf(buf, 64, "track_%06lu.bin", evnum);
        buf[63] = 0;
        filename += buf;
    }
    std::ofstream outfile (filename.c_str(), std::ofstream::binary);
    outfile.write(reinterpret_cast<char*>(data.data()), data.size());
}

// vim: sw=4:tw=78:ft=cpp:et
