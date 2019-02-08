#include <cstdio>

#include "TrackChecker.h"

void TrackCheckerVelo::SetCategories()
{
  m_categories = {
    {// define which categories to monitor
     TrackEffReport({
       "Electrons long eta25",
       [](MCParticles::const_reference& mcp) { return mcp.isLong && mcp.isElectron() && mcp.inEta2_5(); },
     }),
     TrackEffReport({
       "Electrons long fromB eta25",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Electrons long fromB eta25 p<5GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3;
       },
     }),
     TrackEffReport({
       "Electrons long fromB eta25 p>3GeV pt>400MeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3 && mcp.pt > 400;
       },
     }),
     TrackEffReport({
       "Electrons long fromB eta25 p>5GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3;
       },
     }),
     TrackEffReport({
       "Electrons long fromD eta25",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Electrons long fromD eta25 p<5GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3;
       },
     }),
     TrackEffReport({
       "Electrons long fromD eta25 p>3GeV pt>400MeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3 && mcp.pt > 400;
       },
     }),
     TrackEffReport({
       "Electrons long fromD eta25 p>5GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3;
       },
     }),
     TrackEffReport({
       "Electrons long eta25 p<5GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3;
       },
     }),
     TrackEffReport({
       "Electrons long eta25 p>3GeV pt>400MeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3 && mcp.pt > 400;
       },
     }),
     TrackEffReport({
       "Electrons long eta25 p>5GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3;
       },
     }),
     TrackEffReport({
       "Electrons long strange eta25",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromStrangeDecay && mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Electrons long strange eta25 p<5GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromStrangeDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3;
       },
     }),
     TrackEffReport({
       "Electrons long strange eta25 p>3GeV pt>400MeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromStrangeDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3 && mcp.pt > 400;
       },
     }),
     TrackEffReport({
       "Electrons long strange eta25 p>5GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromStrangeDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3;
       },
     }),
     TrackEffReport({
       "Electrons Velo",
       [](MCParticles::const_reference& mcp) { return mcp.hasVelo && mcp.isElectron(); },
     }),
     TrackEffReport({
       "Electrons Velo backward",
       [](MCParticles::const_reference& mcp) { return mcp.hasVelo && mcp.isElectron() && mcp.eta < 0; },
     }),
     TrackEffReport({
       "Electrons Velo forward",
       [](MCParticles::const_reference& mcp) { return mcp.hasVelo && mcp.isElectron() && mcp.eta > 0; },
     }),
     TrackEffReport({
       "Electrons Velo eta25",
       [](MCParticles::const_reference& mcp) { return mcp.hasVelo && mcp.isElectron() && mcp.inEta2_5(); },
     }),
     TrackEffReport({
       "Not electron long eta25",
       [](MCParticles::const_reference& mcp) { return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5(); },
     }),
     TrackEffReport({
       "Not electron long fromB eta25",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Not electron long fromB eta25 p<5GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3;
       },
     }),
     TrackEffReport({
       "Not electron long fromB eta25 p>3GeV pt>400MeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3 && mcp.pt > 400;
       },
     }),
     TrackEffReport({
       "Not electron long fromB eta25 p>5GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3;
       },
     }),
     TrackEffReport({
       "Not electron long fromD eta25",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Not electron long fromD eta25 p<5GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3;
       },
     }),
     TrackEffReport({
       "Not electron long fromD eta25 p>3GeV pt>400MeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3 && mcp.pt > 400;
       },
     }),
     TrackEffReport({
       "Not electron long fromD eta25 p>5GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3;
       },
     }),
     TrackEffReport({
       "Not electron long eta25 p<5GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3;
       },
     }),
     TrackEffReport({
       "Not electron long eta25 p>3GeV pt>400MeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3 && mcp.pt > 400;
       },
     }),
     TrackEffReport({
       "Not electron long eta25 p>5GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3;
       },
     }),
     TrackEffReport({
       "Not electron long strange eta25",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Not electron long strange eta25 p<5GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3;
       },
     }),
     TrackEffReport({
       "Not electron long strange eta25 p>3GeV pt>400MeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3 &&
                mcp.pt > 400;
       },
     }),
     TrackEffReport({
       "Not electron long strange eta25 p>5GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3;
       },
     }),
     TrackEffReport({
       "Not electron Velo",
       [](MCParticles::const_reference& mcp) { return mcp.hasVelo && !mcp.isElectron(); },
     }),
     TrackEffReport({
       "Not electron Velo backward",
       [](MCParticles::const_reference& mcp) { return mcp.hasVelo && !mcp.isElectron() && mcp.eta < 0; },
     }),
     TrackEffReport({
       "Not electron Velo forward",
       [](MCParticles::const_reference& mcp) { return mcp.hasVelo && !mcp.isElectron() && mcp.eta > 0; },
     }),
     TrackEffReport({
       "Not electron Velo eta25",
       [](MCParticles::const_reference& mcp) { return mcp.hasVelo && !mcp.isElectron() && mcp.inEta2_5(); },
     })}};
};

void TrackCheckerVelo::SetHistoCategories()
{
  m_histo_categories = {
    {// define which categories to create histograms for
     HistoCategory({
       "VeloTracks_electrons",
       [](MCParticles::const_reference& mcp) { return mcp.hasVelo && mcp.isElectron(); },
     }),
     HistoCategory({
       "VeloTracks_eta25_electrons",
       [](MCParticles::const_reference& mcp) { return mcp.hasVelo && mcp.isElectron() && mcp.inEta2_5(); },
     }),
     HistoCategory({
       "LongFromB_eta25_electrons",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     HistoCategory({
       "LongFromD_eta25_electrons",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     HistoCategory({
       "LongStrange_eta25_electrons",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromStrangeDecay && mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     HistoCategory({
       "VeloTracks_notElectrons",
       [](MCParticles::const_reference& mcp) { return mcp.hasVelo && !mcp.isElectron(); },
     }),
     HistoCategory({
       "VeloTracks_eta25_notElectrons",
       [](MCParticles::const_reference& mcp) { return mcp.hasVelo && !mcp.isElectron() && mcp.inEta2_5(); },
     }),
     HistoCategory({
       "LongFromB_eta25_notElectrons",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     HistoCategory({
       "LongFromD_eta25_notElectrons",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     HistoCategory({
       "LongStrange_eta25_notElectrons",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5();
       },
     })}};
};

void TrackCheckerVeloUT::SetCategories()
{
  m_categories = {
    {// define which categories to monitor
     TrackEffReport({
       "Velo",
       [](MCParticles::const_reference& mcp) { return mcp.hasVelo && !mcp.isElectron() && mcp.inEta2_5(); },
     }),
     TrackEffReport({
       "Velo+UT",
       [](MCParticles::const_reference& mcp) {
         return mcp.hasVelo && mcp.hasUT && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Velo+UT, p > 5 GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.hasVelo && mcp.hasUT && mcp.p > 5e3 && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Velo, not long",
       [](MCParticles::const_reference& mcp) {
         return mcp.hasVelo && !mcp.isLong && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Velo+UT, not long",
       [](MCParticles::const_reference& mcp) {
         return mcp.hasVelo && mcp.hasUT && !mcp.isLong && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Velo+UT, not long, p > 5 GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.hasVelo && mcp.hasUT && !mcp.isLong && mcp.p > 5e3 && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Long",
       [](MCParticles::const_reference& mcp) { return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5(); },
     }),
     TrackEffReport({
       "Long, p > 5 GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.p > 5e3 && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Long from B",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Long from B, p > 5 GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && mcp.p > 5e3 && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Long electrons",
       [](MCParticles::const_reference& mcp) { return mcp.isLong && mcp.isElectron() && mcp.inEta2_5(); },
     }),
     TrackEffReport({
       "Long from B electrons",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Long from B electrons, p > 5 GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.p > 5e3 && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Long from B, p > 3 GeV, pt > 0.5 GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.p > 3e3 && mcp.pt > 0.5e3 &&
                mcp.inEta2_5();
       },
     })}};
};

void TrackCheckerVeloUT::SetHistoCategories()
{
  m_histo_categories = {
    {// define which categories to create histograms for
     HistoCategory({
       "VeloUTTracks_eta25_electrons",
       [](MCParticles::const_reference& mcp) { return mcp.hasVelo && mcp.hasUT && mcp.isElectron() && mcp.inEta2_5(); },
     }),
     HistoCategory({
       "LongFromB_eta25_electrons",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     HistoCategory({
       "LongFromD_eta25_electrons",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     HistoCategory({
       "LongStrange_eta25_electrons",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromStrangeDecay && mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     HistoCategory({
       "VeloUTTracks_eta25_notElectrons",
       [](MCParticles::const_reference& mcp) {
         return mcp.hasVelo && mcp.hasUT && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     HistoCategory({
       "LongFromB_eta25_notElectrons",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     HistoCategory({
       "LongFromD_eta25_notElectrons",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     HistoCategory({
       "LongStrange_eta25_notElectrons",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5();
       },
     })}};
};

void TrackCheckerForward::SetCategories()
{
  m_categories = {
    {// define which categories to monitor
     TrackEffReport({
       "Long",
       [](MCParticles::const_reference& mcp) { return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5(); },
     }),
     TrackEffReport({
       "Long, p > 5 GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.p > 5e3 && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Long strange",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Long strange, p > 5 GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.p > 5e3 && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Long from B",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Long from B, p > 5 GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && mcp.p > 5e3 && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Long electrons",
       [](MCParticles::const_reference& mcp) { return mcp.isLong && mcp.isElectron() && mcp.inEta2_5(); },
     }),
     TrackEffReport({
       "Long electrons from B",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Long electrons from B, p > 5 GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && mcp.p > 5e3 && mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     TrackEffReport({
       "Long from B, p > 3 GeV, pt > 0.5 GeV",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.p > 3e3 && mcp.pt > 0.5e3 &&
                mcp.inEta2_5();
       },
     })}};
};

void TrackCheckerForward::SetHistoCategories()
{
  m_histo_categories = {
    {// define which categories to create histograms for
     HistoCategory({
       "Long_eta25_electrons",
       [](MCParticles::const_reference& mcp) { return mcp.isLong && mcp.isElectron() && mcp.inEta2_5(); },
     }),
     HistoCategory({
       "LongFromB_eta25_electrons",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     HistoCategory({
       "LongFromD_eta25_electrons",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     HistoCategory({
       "LongStrange_eta25_electrons",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromStrangeDecay && mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     HistoCategory({
       "Long_eta25_notElectrons",
       [](MCParticles::const_reference& mcp) { return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5(); },
     }),
     HistoCategory({
       "LongFromB_eta25_notElectrons",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     HistoCategory({
       "LongFromD_eta25_notElectrons",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5();
       },
     }),
     HistoCategory({
       "LongStrange_eta25_notElectrons",
       [](MCParticles::const_reference& mcp) {
         return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5();
       },
     })}};
};
