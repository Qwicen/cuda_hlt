#pragma once

#include "Common.h"
#include "InputTools.h"
#include "PV_Definitions.cuh"
#include "patPV_Definitions.cuh"

#include <algorithm>

#ifdef WITH_ROOT
#include "TFile.h"
#include "TH1D.h"
#include "TTree.h"
#endif

// configuration for PV checker -> check values
static constexpr int nTracksToBeRecble = 4;
static constexpr float dzIsolated = 10.f; // mm
static constexpr bool matchByTracks = false;

void checkPVs(
  const std::string& foldername,
  uint number_of_files,
  PV::Vertex* rec_vertex,
  int* number_of_vertex,
  const uint number_of_selected_events,
  const uint* event_list,
  const std::string mode);

struct MCVertex {
  double x;
  double y;
  double z;
  int numberTracks;
};

typedef struct {
  MCVertex* pMCPV;          // pointer to MC PV
  int nRecTracks;           // number of reconstructed tracks from this MCPV
  int nRecBackTracks;       // number of reconstructed backward tracks
  int indexRecPVInfo;       // index to reconstructed PVInfo (-1 if not reco)
  int nCorrectTracks;       // correct tracks belonging to reconstructed PV
  int multClosestMCPV;      // multiplicity of closest reconstructable MCPV
  double distToClosestMCPV; // distance to closest reconstructible MCPV
  int decayCharm;           // type of mother particle
  int decayBeauty;
  // std::vector<LHCb::MCParticle*> m_mcPartInMCPV;
  // std::vector<LHCb::Track*> m_recTracksInMCPV;
} MCPVInfo;

typedef struct {
public:
  int nTracks;     // number of tracks
  int nVeloTracks; // number of velo tracks in a vertex
  int nLongTracks;
  double minTrackRD; //
  double maxTrackRD; //
  double chi2;
  double nDoF;
  double d0;
  double d0nTr;
  double chi2nTr;
  double mind0;
  double maxd0;
  int mother;
  // XYZPoint position;       // position
  double x;
  double y;
  double z;
  PatPV::XYZPoint positionSigma; // position sigmas
  int indexMCPVInfo;             // index to MCPVInfo
  PV::Vertex* pRECPV;            // pointer to REC PV
} RecPVInfo;

void match_mc_vertex_by_distance(int ipv, std::vector<RecPVInfo>& rinfo, std::vector<MCPVInfo>& mcpvvec)
{

  double mindist = 999999.;
  int indexmc = -1;

  for (int imc = 0; imc < (int) mcpvvec.size(); imc++) {
    if (mcpvvec[imc].indexRecPVInfo > -1) continue;
    double dist = fabs(mcpvvec[imc].pMCPV->z - rinfo[ipv].z);
    if (dist < mindist) {
      mindist = dist;
      indexmc = imc;
    }
  }
  if (indexmc > -1) {
    if (mindist < 5.0 * rinfo[ipv].positionSigma.z) {
      rinfo[ipv].indexMCPVInfo = indexmc;
      mcpvvec[indexmc].indexRecPVInfo = ipv;
    }
  }
}

void printRat(std::string mes, int a, int b)
{

  float rat = 0.f;
  if (b > 0) rat = 1.0f * a / b;

  // reformat message
  unsigned int len = 20;
  std::string pmes = mes;
  while (pmes.length() < len) {
    pmes += " ";
  }
  pmes += " : ";

  info_cout << pmes << " " << rat << "( " << a << " / " << b << " )" << std::endl;
}

std::vector<MCPVInfo>::iterator closestMCPV(std::vector<MCPVInfo>& rblemcpv, std::vector<MCPVInfo>::iterator& itmc)
{

  std::vector<MCPVInfo>::iterator itret = rblemcpv.end();
  double mindist = 999999.;
  if (rblemcpv.size() < 2) return itret;
  std::vector<MCPVInfo>::iterator it;
  for (it = rblemcpv.begin(); it != rblemcpv.end(); it++) {
    if (it->pMCPV != itmc->pMCPV) {
      double diff_x = it->pMCPV->x - itmc->pMCPV->x;
      double diff_y = it->pMCPV->y - itmc->pMCPV->y;
      double diff_z = it->pMCPV->z - itmc->pMCPV->z;
      double dist = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

      if (dist < mindist) {
        mindist = dist;
        itret = it;
      }
    }
  }
  return itret;
}
