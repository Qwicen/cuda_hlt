/** @file TrackChecker.cc
 *
 * @brief check tracks against MC truth
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-19
 */

#include <cstdio>

#include "TrackChecker.h"

TrackChecker::TrackChecker() :
    m_categories({ // define which categories to monitor
        TrackEffReport({ "Velo",
                [] (const MCParticles::const_reference& mcp)
                { return mcp.isVelo() && 11 != std::abs(mcp.pid()); },
                }),
        TrackEffReport({ "Velo, p > 5 GeV",
                [] (const MCParticles::const_reference& mcp)
                { return mcp.isVelo() && 11 != std::abs(mcp.pid()) && mcp.p() > 5e3; },
                }),
        TrackEffReport({ "Long",
                [] (const MCParticles::const_reference& mcp)
                { return mcp.isLong() && 11 != std::abs(mcp.pid()); },
                }),
        TrackEffReport({ "Long, p > 5 GeV",
                [] (const MCParticles::const_reference& mcp)
                { return mcp.isLong() && 11 != std::abs(mcp.pid()) && mcp.p() > 5e3; },
                }),
        TrackEffReport({ "Long from B",
                [] (const MCParticles::const_reference& mcp)
                { return mcp.isLong() && mcp.isFromB() && 11 != std::abs(mcp.pid()); },
                }),
        TrackEffReport({ "Long from B, p > 5 GeV",
                [] (const MCParticles::const_reference& mcp)
                { return mcp.isLong() && mcp.isFromB() && 11 != std::abs(mcp.pid()) && mcp.p() > 5e3; },
                }),
        TrackEffReport({ "Long from D",
                [] (const MCParticles::const_reference& mcp)
                { return mcp.isLong() && mcp.isFromD() && 11 != std::abs(mcp.pid()); },
                }),
        TrackEffReport({ "Long from D, p > 5 GeV",
                [] (const MCParticles::const_reference& mcp)
                { return mcp.isLong() && mcp.isFromD() && 11 != std::abs(mcp.pid()) && mcp.p() > 5e3; },
                }),
        TrackEffReport({ "Long strange",
                [] (const MCParticles::const_reference& mcp)
                { return mcp.isStrangeLong() && 11 != std::abs(mcp.pid()); },
                }),
        TrackEffReport({ "Long strange, p > 5 GeV",
                [] (const MCParticles::const_reference& mcp)
                { return mcp.isStrangeLong() && 11 != std::abs(mcp.pid()) && mcp.p() > 5e3; },
                }),
        TrackEffReport({ "Down",
                [] (const MCParticles::const_reference& mcp)
                { return mcp.isDown() && 11 != std::abs(mcp.pid()); },
                }),
        TrackEffReport({ "Down, p > 5 GeV",
                [] (const MCParticles::const_reference& mcp)
                { return mcp.isDown() && 11 != std::abs(mcp.pid()) && mcp.p() > 5e3; },
                }),
        TrackEffReport({ "Down strange",
                [] (const MCParticles::const_reference& mcp)
                { return mcp.isStrangeDown() && 11 != std::abs(mcp.pid()); },
                }),
        TrackEffReport({ "Down strange, p > 5 GeV",
                [] (const MCParticles::const_reference& mcp)
                { return mcp.isStrangeDown() && 11 != std::abs(mcp.pid()) && mcp.p() > 5e3; },
                }),
        })
{}

TrackChecker::~TrackChecker()
{
    std::printf("\n");
    std::printf("%-23s: %9lu/%9lu %6.2f%% (%6.2f%%) ghosts\n",
            "TrackChecker output",
            m_nghosts, m_ntracks,
            100.f * float(m_nghosts) / float(m_ntracks),
            100.f * m_ghostperevent);
    m_categories.clear();
    std::printf("\n");
}

void TrackChecker::TrackEffReport::operator()(const MCParticles& mcps)
{
    for (auto mcp: mcps) {
        if (m_accept(mcp)) {
            ++m_naccept, ++m_nacceptperevt;
        }
    }
}

void TrackChecker::TrackEffReport::operator()(
        const Tracks::const_reference& track,
        const MCParticles::const_reference& mcp,
        const float weight)
{
    if (!m_accept(mcp)) return;
    if (!m_keysseen.count(mcp.key())) {
        ++m_nfound, ++m_nfoundperevt;
        m_keysseen.insert(mcp.key());
        // update purity
        m_hitpur *= float(m_nfound - 1) / float(m_nfound);
        m_hitpur += weight / float(m_nfound);
        // update hit efficiency
        auto hiteff = track.nIDs() * weight / float(mcp.nIDs());
        m_hiteff *= float(m_nfound - 1) / float(m_nfound);
        m_hiteff += hiteff / float(m_nfound);
    } else {
        // clone, update total hit efficiency by adding what's missing to
        // m_hiteff
        auto hiteff = track.nIDs() * weight / float(mcp.nIDs());
        m_hiteff += hiteff / float(m_nfound);
        ++m_nclones;
    }
}

void TrackChecker::TrackEffReport::evtEnds()
{
    m_keysseen.clear();
    if (m_nacceptperevt) {
        m_effperevt *= float(m_nevents) / float(m_nevents + 1);
        ++m_nevents;
        m_effperevt += (float(m_nfoundperevt) / float(m_nacceptperevt)) / float(m_nevents);
    }
    m_nfoundperevt = m_nacceptperevt = 0;
}

TrackChecker::TrackEffReport::~TrackEffReport()
{
    auto clonerate = 0.f, eff = 0.f;
    if (m_nfound) clonerate = float(m_nclones) / float(m_nfound + m_nfound);
    if (m_naccept) eff = float(m_nfound) / float(m_naccept);
    std::printf("%-23s: %9lu/%9lu %6.2f%% (%6.2f%%), "
            "%9lu (%6.2f%%) clones, hit eff %6.2f%% pur %6.2f%%\n",
            m_name.c_str(), m_nfound, m_naccept,
            100.f * eff,
            100.f * m_effperevt, m_nclones,
            100.f * clonerate,
            100.f * m_hiteff, 100.f * m_hitpur);
}

void TrackChecker::operator()(const Tracks& tracks,
        const MCAssociator& mcassoc, const MCParticles& mcps)
{
    // register MC particles
    for (auto& report: m_categories) report(mcps);
    // go through tracks
    const std::size_t ntracksperevt = tracks.size();
    std::size_t nghostsperevt = 0;
    for (auto track: tracks) {
        // check LHCbIDs for MC association
        const auto& ids = track.ids();
        const auto assoc = mcassoc(ids.begin(), ids.end());
        if (!assoc) {
            ++nghostsperevt;
            continue;
        }
        // have MC association, check weight
        const auto weight = assoc.front().second;
        if (weight < m_minweight) {
            ++nghostsperevt;
            continue;
        }
        // okay, sufficient to proceed...
        const auto mcp = assoc.front().first;
        // add to various categories
        for (auto& report: m_categories) report(track, mcp, weight);
    }
    // almost done, notify of end of event...
    ++m_nevents;
    for (auto& report: m_categories) report.evtEnds();
    m_ghostperevent *= float(m_nevents - 1) / float(m_nevents);
    if (ntracksperevt) {
        m_ghostperevent += (float(nghostsperevt) / float(ntracksperevt)) / float(m_nevents);
    }
    m_nghosts += nghostsperevt, m_ntracks += ntracksperevt;
}

// vim: sw=4:tw=78:ft=cpp:et