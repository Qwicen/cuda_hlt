/** @file TrackChecker.h
 *
 * @brief check tracks against MC truth
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-19
 *
 * 2018-07 Dorothea vom Bruch: updated to run over different track types, 
 * euse exact same categories as PrChecker2, 
 * take input from Renato Quagliani's TrackerDumper
 */

#pragma once

#include <set>
#include <string>
#include <vector>
#include <functional>

#include "Tracks.h"
#include "MCAssociator.h"
#include "Logger.h"

class TrackChecker
{
   protected:
        using AcceptFn = std::function<bool (MCParticles::const_reference&)>;
        struct TrackEffReport {
            std::string m_name;
            AcceptFn m_accept;
            std::size_t m_naccept = 0;
            std::size_t m_nfound = 0;
            std::size_t m_nacceptperevt = 0;
            std::size_t m_nfoundperevt = 0;
            std::size_t m_nclones = 0;
            std::size_t m_nevents = 0;
            float m_effperevt = 0.f;
            float m_hitpur = 0.f;
            float m_hiteff = 0.f;
            std::set<uint32_t> m_keysseen;

            /// no default construction
            TrackEffReport() = delete;
            /// usual copy construction
            TrackEffReport(const TrackEffReport&) = default;
            /// usual move construction
            TrackEffReport(TrackEffReport&&) = default;
            /// usual copy assignment
            TrackEffReport& operator=(const TrackEffReport&) = default;
            /// usual move assignment
            TrackEffReport& operator=(TrackEffReport&&) = default;
            /// construction from name and accept criterion for eff. denom.
            template <typename F>
            TrackEffReport(const std::string& name, const F& accept) :
                m_name(name), m_accept(accept)
            {}
            /// construction from name and accept criterion for eff. denom.
            template <typename F>
            TrackEffReport(std::string&& name, F&& accept) :
                m_name(std::move(name)), m_accept(std::move(accept))
            {}
            /// register MC particles
            void operator()(const MCParticles& mcps);
            /// register track and its MC association
	  void operator()(trackChecker::Tracks::const_reference& track,
                    MCParticles::const_reference& mcp,
                    const float weight);
            /// notify of end of event
            void evtEnds();
            /// free resources, and print result
            ~TrackEffReport();
        };

        const float m_minweight = 0.7f;
        std::vector<TrackEffReport> m_categories;

        std::size_t m_nevents = 0;
        std::size_t m_ntracks = 0;
        std::size_t m_nghosts = 0;
        float m_ghostperevent = 0.f;

    public:
        TrackChecker() {};
        ~TrackChecker();
        void operator()(const trackChecker::Tracks& tracks,
                const MCAssociator& mcassoc,
                const MCParticles& mcps);
};

class TrackCheckerVelo : public TrackChecker
{
  public:
      void SetCategories();
      TrackCheckerVelo() {
        SetCategories();
      };
  
};

class TrackCheckerVeloUT : public TrackChecker
{
  public:
      void SetCategories();
      TrackCheckerVeloUT() {
        SetCategories();
      };
  
};

class TrackCheckerForward : public TrackChecker
{
  public:
      void SetCategories();
      TrackCheckerForward() {
        SetCategories();
      };
  
}; 
