/** @file TrackChecker.cpp
 *
 * @brief check tracks against MC truth
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-19
 *
 * 2018-07 Dorothea vom Bruch: updated to run over different track types, 
 * use exact same categories as PrChecker2, 
 * take input from Renato Quagliani's TrackerDumper
 */

#include <cstdio>

#include "TrackChecker.h"

void TrackCheckerVelo::SetCategories() {
  m_categories = {{ // define which categories to monitor
    // Renato's categories
     TrackEffReport({ "Electrons long eta25",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.isElectron() && mcp.inEta2_5(); },
        }),
     TrackEffReport({ "Electrons long fromB eta25",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5(); },
        }),
     TrackEffReport({ "Electrons long fromB eta25 p<5GeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5() &&  mcp.p < 5e3; },
        }),
     TrackEffReport({ "Electrons long fromB eta25 p>3GeV pt>400MeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5() &&  mcp.p > 3e3 && mcp.pt > 400; },
        }),
     TrackEffReport({ "Electrons long fromB eta25 p>5GeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5() &&  mcp.p > 5e3; },
        }),
     TrackEffReport({ "Electrons long fromD eta25",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5(); },
        }),
     TrackEffReport({ "Electrons long fromD eta25 p<5GeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3; },
        }),
     TrackEffReport({ "Electrons long fromD eta25 p>3GeV pt>400MeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3 && mcp.pt > 400; },
        }),
     TrackEffReport({ "Electrons long fromD eta25 p>5GeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3; },
        }),
     TrackEffReport({ "Electrons long eta25 p<5GeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3; },
        }),
     TrackEffReport({ "Electrons long eta25 p>3GeV pt>400MeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3 && mcp.pt > 400; },
        }),
     TrackEffReport({ "Electrons long eta25 p>5GeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3; },
        }),
     TrackEffReport({ "Electrons long strange eta25",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromStrangeDecay && mcp.isElectron() && mcp.inEta2_5(); },
        }),
     TrackEffReport({ "Electrons long strange eta25 p<5GeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromStrangeDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3; },
        }),
     TrackEffReport({ "Electrons long strange eta25 p>3GeV pt>400MeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromStrangeDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3 && mcp.pt > 400; },
        }),
     TrackEffReport({ "Electrons long strange eta25 p>5GeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromStrangeDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3; },
        }),
     TrackEffReport({ "Electrons Velo",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.hasVelo && mcp.isElectron(); },
        }),
     TrackEffReport({ "Electrons Velo backward",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.hasVelo && mcp.isElectron() && mcp.eta < 0; },
        }),
     TrackEffReport({ "Electrons Velo forward",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.hasVelo && mcp.isElectron() && mcp.eta > 0; },
        }),
     TrackEffReport({ "Electrons Velo eta25",
        [] (const MCParticles::const_reference& mcp)
          { return mcp.hasVelo && mcp.isElectron() && mcp.inEta2_5(); },
        }), 
     TrackEffReport({ "Not electron long eta25",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5(); },
        }),
     TrackEffReport({ "Not electron long fromB eta25",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5(); },
        }),
     TrackEffReport({ "Not electron long fromB eta25 p<5GeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5() &&  mcp.p < 5e3; },
        }),
     TrackEffReport({ "Not electron long fromB eta25 p>3GeV pt>400MeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5() &&  mcp.p > 3e3 && mcp.pt > 400; },
        }),
     TrackEffReport({ "Not electron long fromB eta25 p>5GeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5() &&  mcp.p > 5e3; },
        }),
     TrackEffReport({ "Not electron long fromD eta25",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5(); },
        }),
     TrackEffReport({ "Not electron long fromD eta25 p<5GeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3; },
        }),
     TrackEffReport({ "Not electron long fromD eta25 p>3GeV pt>400MeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3 && mcp.pt > 400; },
        }),
     TrackEffReport({ "Not electron long fromD eta25 p>5GeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3; },
        }),
     TrackEffReport({ "Not electron long eta25 p<5GeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3; },
        }),
     TrackEffReport({ "Not electron long eta25 p>3GeV pt>400MeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3 && mcp.pt > 400; },
        }),
     TrackEffReport({ "Not electron long eta25 p>5GeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3; },
        }),
     TrackEffReport({ "Not electron long strange eta25",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5(); },
        }),
     TrackEffReport({ "Not electron long strange eta25 p<5GeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3; },
        }),
     TrackEffReport({ "Not electron long strange eta25 p>3GeV pt>400MeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3 && mcp.pt > 400; },
        }),
     TrackEffReport({ "Not electron long strange eta25 p>5GeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3; },
        }),
     TrackEffReport({ "Not electron Velo",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.hasVelo && !mcp.isElectron(); },
        }),
     TrackEffReport({ "Not electron Velo backward",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.hasVelo && !mcp.isElectron() && mcp.eta < 0; },
        }),
     TrackEffReport({ "Not electron Velo forward",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.hasVelo && !mcp.isElectron() && mcp.eta > 0; },
        }),
     TrackEffReport({ "Not electron Velo eta25",
        [] (const MCParticles::const_reference& mcp)
          { return mcp.hasVelo && !mcp.isElectron() && mcp.inEta2_5(); },
        })
     

    
    // currently implemented in PrChecker2 (master branch)
    // TrackEffReport({ "Velo",
    //     [] (const MCParticles::const_reference& mcp)
    //     { return mcp.hasVelo && !mcp.isElectron() && mcp.inEta2_5(); },
    //     }),
    // TrackEffReport({ "Long",
    //     [] (const MCParticles::const_reference& mcp)
    //     { return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5(); },
    //     }),
    // TrackEffReport({ "Long, p > 5 GeV",
    //     [] (const MCParticles::const_reference& mcp)
    //     { return mcp.isLong && !mcp.isElectron() && mcp.p > 5e3 && mcp.inEta2_5(); },
    //     }),
    // TrackEffReport({ "Long strange",
    //     [] (const MCParticles::const_reference& mcp)
    //     { return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5(); },
    //     }),
    // TrackEffReport({ "Long strange, p > 5 GeV",
    //     [] (const MCParticles::const_reference& mcp)
    //     { return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.p > 5e3 && mcp.inEta2_5(); },
    //     }),
    // TrackEffReport({ "Long from B",
    //     [] (const MCParticles::const_reference& mcp)
    //     { return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5(); },
    //     }),
    // TrackEffReport({ "Long from B, p > 5 GeV",
    //     [] (const MCParticles::const_reference& mcp)
    //     { return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.p > 5e3 && mcp.inEta2_5(); },
    //       }),
    // TrackEffReport({ "Long electrons",
    //     [] (const MCParticles::const_reference& mcp)
    //     { return mcp.isLong && mcp.isElectron() && mcp.inEta2_5(); },
    //       }),
    // TrackEffReport({ "Long from B electrons",
    //     [] (const MCParticles::const_reference& mcp)
    //     { return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5(); },
    //       }),
    // TrackEffReport({ "Long from B electrons, p > 5 GeV",
    //     [] (const MCParticles::const_reference& mcp)
    //     { return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.p > 5e3 && mcp.inEta2_5(); },
    //       }),
    // TrackEffReport({ "Long from B, p > 3 GeV, pt > 0.5 GeV",
    //     [] (const MCParticles::const_reference& mcp)
    //     { return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.p > 3e3 && mcp.pt > 0.5e3 && mcp.inEta2_5(); },
    //       })



    
    // TrackEffReport({ "Long from D",
    //     [] (const MCParticles::const_reference& mcp)
    //     { return mcp.isLong && mcp.isFromD && !mcp.isElectron() && mcp.inEta2_5(); },
    //     }),
    // TrackEffReport({ "Long from D, p > 5 GeV",
    //     [] (const MCParticles::const_reference& mcp)
    //     { return mcp.isLong && mcp.isFromD && !mcp.isElectron() && mcp.p > 5e3 && mcp.inEta2_5(); },
    //     }),
    // TrackEffReport({ "Down",
    //     [] (const MCParticles::const_reference& mcp)
    //     { return mcp.isDown && !mcp.isElectron() && mcp.inEta2_5(); },
    //     }),
    // TrackEffReport({ "Down, p > 5 GeV",
    //     [] (const MCParticles::const_reference& mcp)
    //     { return mcp.isDown && !mcp.isElectron() && mcp.p > 5e3 && mcp.inEta2_5(); },
    //     }),
    // TrackEffReport({ "Down strange",
    //     [] (const MCParticles::const_reference& mcp)
    //     { return mcp.isStrangeDown && !mcp.isElectron() && mcp.inEta2_5(); },
    //     }),
    // TrackEffReport({ "Down strange, p > 5 GeV",
    //     [] (const MCParticles::const_reference& mcp)
    //     { return mcp.isStrangeDown && !mcp.isElectron() && mcp.p > 5e3 && mcp.inEta2_5(); },
    // 	})
    }};
};
 

void TrackCheckerVeloUT::SetCategories() {
  m_categories = {{ // define which categories to monitor
    TrackEffReport({ "Velo",
        [] (const MCParticles::const_reference& mcp)
	  { return mcp.hasVelo && !mcp.isElectron() && mcp.inEta2_5(); },
        }),
    TrackEffReport({ "Velo+UT",
        [] (const MCParticles::const_reference& mcp)
	  { return mcp.hasVelo && mcp.hasUT && !mcp.isElectron() && mcp.inEta2_5(); },
        }),
    TrackEffReport({ "Velo+UT, p > 5 GeV",
	[] (const MCParticles::const_reference& mcp)
	  { return mcp.hasVelo && mcp.hasUT && mcp.p > 5e3 && !mcp.isElectron() && mcp.inEta2_5(); },
	}),
    TrackEffReport({ "Velo, not long",
        [] (const MCParticles::const_reference& mcp)
	  { return mcp.hasVelo && !mcp.isLong && !mcp.isElectron() && mcp.inEta2_5(); },
        }),
    TrackEffReport({ "Velo+UT, not long",
        [] (const MCParticles::const_reference& mcp)
	  { return mcp.hasVelo && mcp.hasUT && !mcp.isLong && !mcp.isElectron() && mcp.inEta2_5(); },
	  }),
    TrackEffReport({ "Velo+UT, not long, p > 5 GeV",
        [] (const MCParticles::const_reference& mcp)
	  { return mcp.hasVelo && mcp.hasUT && !mcp.isLong && mcp.p > 5e3 && !mcp.isElectron() && mcp.inEta2_5(); },
	  }),
    TrackEffReport({ "Long",
	  [] (const MCParticles::const_reference& mcp)
	    { return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5(); },
	  }),
    TrackEffReport({ "Long, p > 5 GeV",
	  [] (const MCParticles::const_reference& mcp)
	    { return mcp.isLong && mcp.p > 5e3 && !mcp.isElectron() && mcp.inEta2_5(); },
	  }),
    TrackEffReport({ "Long from B",
	  [] (const MCParticles::const_reference& mcp)
	    { return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5(); },
	  }),
    TrackEffReport({ "Long from B, p > 5 GeV",
	  [] (const MCParticles::const_reference& mcp)
	    { return mcp.isLong && mcp.fromBeautyDecay && mcp.p > 5e3 && !mcp.isElectron() && mcp.inEta2_5(); },
	  }),
    TrackEffReport({ "Long electrons",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.isElectron() && mcp.inEta2_5(); },
          }),
    TrackEffReport({ "Long from B electrons",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5(); },
          }),
    TrackEffReport({ "Long from B electrons, p > 5 GeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.p > 5e3 && mcp.inEta2_5(); },
          }),
    TrackEffReport({ "Long from B, p > 3 GeV, pt > 0.5 GeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.p > 3e3 && mcp.pt > 0.5e3 && mcp.inEta2_5(); },
          })
    }}; 
}; 


void TrackCheckerForward::SetCategories() {
  m_categories = {{ // define which categories to monitor
      TrackEffReport({ "Long",
	  [] (const MCParticles::const_reference& mcp)
	    { return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5(); },
	  }),
    TrackEffReport({ "Long, p > 5 GeV",
	  [] (const MCParticles::const_reference& mcp)
	    { return mcp.isLong && mcp.p > 5e3 && !mcp.isElectron() && mcp.inEta2_5(); },
	  }),
    TrackEffReport({ "Long strange",
          [] (const MCParticles::const_reference& mcp)
            { return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5(); },
          }),
    TrackEffReport({ "Long strange, p > 5 GeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.p > 5e3 && mcp.inEta2_5(); },
        }),
    TrackEffReport({ "Long from B",
	  [] (const MCParticles::const_reference& mcp)
	    { return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5(); },
	  }),
    TrackEffReport({ "Long from B, p > 5 GeV",
	  [] (const MCParticles::const_reference& mcp)
	    { return mcp.isLong && mcp.fromBeautyDecay && mcp.p > 5e3 && !mcp.isElectron() && mcp.inEta2_5(); },
	  }),
    TrackEffReport({ "Long electrons",
	  [] (const MCParticles::const_reference& mcp)
	    { return mcp.isLong && mcp.isElectron() && mcp.inEta2_5(); },
	  }),
    TrackEffReport({ "Long electrons from B",
	  [] (const MCParticles::const_reference& mcp)
	    { return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5(); },
	  }),
    TrackEffReport({ "Long electrons from B, p > 5 GeV",
	  [] (const MCParticles::const_reference& mcp)
	    { return mcp.isLong && mcp.fromBeautyDecay && mcp.p > 5e3 && mcp.isElectron() && mcp.inEta2_5(); },
	  }),
    TrackEffReport({ "Long from B, p > 3 GeV, pt > 0.5 GeV",
        [] (const MCParticles::const_reference& mcp)
        { return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.p > 3e3 && mcp.pt > 0.5e3 && mcp.inEta2_5(); },
          })
    }}; 
};   

TrackChecker::~TrackChecker()
{
  std::printf("%-50s: %9lu/%9lu %6.2f%% (%6.2f%%) ghosts\n",
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
					    const trackChecker::Tracks::const_reference& track,
					    const MCParticles::const_reference& mcp,
					    const float weight)
{

  if (!m_accept(mcp)) return;
  if (!m_keysseen.count(mcp.key)) {
    ++m_nfound, ++m_nfoundperevt;
    m_keysseen.insert(mcp.key);
  } else {
    ++m_nclones;
  }
  
  // update purity
  m_hitpur *= float(m_nfound + m_nclones - 1) / float(m_nfound + m_nclones);
  m_hitpur += weight / float(m_nfound + m_nclones);
  // update hit efficiency
  //auto hiteff = track.numHits * weight / float(mcp.numHits);
  auto hiteff = track.n_matched_total * weight / float(mcp.numHits);
  m_hiteff *= float(m_nfound + m_nclones - 1) / float(m_nfound + m_nclones);
  m_hiteff += hiteff / float(m_nfound + m_nclones);

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

  if (m_naccept > 0) {
    std::printf("%-50s: %9lu/%9lu %6.2f%% (%6.2f%%), "
        "%9lu (%6.2f%%) clones, hit eff %6.2f%% pur %6.2f%%\n",
        m_name.c_str(), m_nfound, m_naccept,
        100.f * eff,
        100.f * m_effperevt, m_nclones,
        100.f * clonerate,
        100.f * m_hiteff, 100.f * m_hitpur);
  }
}

void TrackChecker::operator()(const trackChecker::Tracks& tracks,
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
    const auto assoc = mcassoc(ids.begin(), ids.end(), track.n_matched_total);
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
    for (auto& report: m_categories) {
      report(track, mcp, weight);
      
    }
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
