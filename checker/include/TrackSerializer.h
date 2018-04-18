/** @file TrackSerializer.h
 *
 * @brief simple binary track serializer
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-06
 */

#pragma once

#include <string>

#include "Tracks.h"

class TrackSerializer {
    private:
        std::string m_pfx;
    public:
        /// construct file names to serialize tracks into from prefix pfx
        TrackSerializer(const std::string& pfx) : m_pfx(pfx) {}

        /// serialize tracks for event number
        void operator()(uint64_t evnum, const Tracks& tracks) const;
};

// vim: sw=4:tw=78:ft=cpp:et
