#pragma once

#include <string>
#include <cstdint>
#include <vector>

#include "VeloPixels.h"
#include "MCParticle.h"

#include "../../../main/include/Common.h"

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

    // Monte Carlo information
    struct MCP {
        uint32_t key;
        uint32_t id;
        float p;
        float pt;
        float eta;
        float phi;
        bool islong;
        bool isdown;
        bool isvelo;
        bool isut;
        bool strangelong;
        bool strangedown;
        bool fromb;
        bool fromd;
        uint32_t numHits;
        std::vector<uint32_t> hits;
    };

    std::vector<MCP> mcps;

    // Constructor
    VelopixEvent(const std::vector<uint8_t>& _event, const bool checkFile = true);

    void print() const;

    /// get hits into a format we like (should ultimately go away, or just be a view)
    VeloPixels soaHits() const;
    MCParticles mcparticles() const;
};

class VelopixEventReader {
private:
    constexpr static int numberOfModules = 52;

public:
    static bool fileExists (const std::string& name);

    static void readFileIntoVector(const std::string& filename, std::vector<uint8_t>& output);

    static std::vector<VelopixEvent> readFolder(
      const std::string& foldername, uint nFiles = 0, const bool checkFiles = true);
};

// vim: sw=4:tw=78:ft=cpp:et
