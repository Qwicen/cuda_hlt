/** @file velopix-input-reader.h
 *
 * @brief a reader of velopix inputs
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-18
 */

#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include <algorithm>

#include "MCParticle.h"

#include "../../../main/include/Common.h"
#include "../../../main/include/Logger.h"

class VelopixEvent {
private:
    template<class T>
    static std::string strVector(const T v, const uint vSize, const uint numberOfElements = 5) {
        std::string s = "";
        auto n = std::min(vSize, numberOfElements);
        for (size_t i=0; i<n; ++i) {
            s += std::to_string(v[i]);
            if (i != n-1) s += ", ";
            else if (i == vSize-1) s += "";
            else s += "...";
        }
        return s;
    }

public:
    uint32_t size;

    // Event data
    uint32_t numberOfModules;
    uint32_t numberOfHits;
    std::vector<float> module_Zs;
    std::vector<uint32_t> module_hitStarts;
    std::vector<uint32_t> module_hitNums;
    std::vector<uint32_t> hit_IDs;
    std::vector<float> hit_Xs;
    std::vector<float> hit_Ys;
    std::vector<float> hit_Zs;
    MCParticles mcps;

    // Constructor
    VelopixEvent(const std::vector<char>& _event, const bool checkFile = true);

    void print() const;

    MCParticles mcparticles() const;
};
