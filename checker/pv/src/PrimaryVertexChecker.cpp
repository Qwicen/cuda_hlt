#include "PrimaryVertexChecker.h"

void checkPVs(
  const std::string& foldername,
  uint number_of_files,
  PV::Vertex* rec_vertex,
  int* number_of_vertex,
  const uint number_of_selected_events,
  const uint* event_list,
  const std::string mode)
{
  std::vector<std::string> folderContents = list_folder(foldername);

  uint requestedFiles = number_of_files == 0 ? folderContents.size() : number_of_files;

  if (requestedFiles > folderContents.size()) {
    warning_cout << "Requested " << requestedFiles << " files, but only " << folderContents.size()
                 << " files are present" << std::endl
                 << std::endl;
  }
  else {
    verbose_cout << "Requested " << requestedFiles << " files" << std::endl;

    int readFiles = 0;

    // vector containing for each event vector of MCVertices
    std::vector<std::vector<MCVertex>> events_vertices;

    // vector containing MC vertices
    // std::vector<MCVertex> vertices;

    // counters for efficiencies/fake rate
    int sum_nMCPV = 0;
    int sum_nRecMCPV = 0;
    int sum_nMCPV_isol = 0;
    int sum_nRecMCPV_isol = 0;
    int sum_nMCPV_close = 0;
    int sum_nRecMCPV_close = 0;
    int sum_nFalsePV = 0;
    int sum_nFalsePV_real = 0;

    // vectors to collect the pulls and erros
    std::vector<double> vec_diff_x;
    std::vector<double> vec_diff_y;
    std::vector<double> vec_diff_z;

    std::vector<double> vec_err_x;
    std::vector<double> vec_err_y;
    std::vector<double> vec_err_z;

    // loop over files/events
    std::vector<std::vector<MCVertex>> MC_vertices_events;
    for (uint i_event = 0; i_event < requestedFiles; ++i_event) {
      // Read event #i in the list and add it to the inputs
      // if more files are requested than present in folder, read them again

      // collect true PV vertices in a event
      std::string readingFile = folderContents[i_event % folderContents.size()];
      std::string filename = foldername + "/" + readingFile;
      std::vector<char> inputContents;
      readFileIntoVector(foldername + "/" + readingFile, inputContents);

      uint8_t* input = (uint8_t*) inputContents.data();

      int number_mcpv = *((int*) input);
      input += sizeof(int);

      std::vector<MCVertex> MC_vertices;
      for (uint32_t i = 0; i < number_mcpv; ++i) {
        MCVertex mc_vertex;

        int VertexNumberOfTracks = *((int*) input);
        input += sizeof(int);
        mc_vertex.numberTracks = VertexNumberOfTracks;
        mc_vertex.x = *((double*) input);
        input += sizeof(double);
        mc_vertex.y = *((double*) input);
        input += sizeof(double);
        mc_vertex.z = *((double*) input);
        input += sizeof(double);

        // if(mc_vertex.numberTracks >= 4) vertices.push_back(mc_vertex);
        MC_vertices.push_back(mc_vertex);
      }
      MC_vertices_events.push_back(MC_vertices);
    }
    // events selected by global event cuts
    std::vector<std::vector<MCVertex>> MC_vertices_selected_events;
    for (int i = 0; i < number_of_selected_events; i++) {
      const uint event = event_list[i];
      std::vector<MCVertex> MC_vertices = MC_vertices_events[event];
      MC_vertices_selected_events.push_back(MC_vertices);
    }

    // loop over selected events
    for (uint i_event = 0; i_event < number_of_selected_events; ++i_event) {
      std::vector<PV::Vertex*> vecOfVertices;
      // first fill vector with vertices
      for (uint i = 0; i < number_of_vertex[i_event]; i++) {
        int index = i_event * PatPV::max_number_vertices + i;
        vecOfVertices.push_back(&(rec_vertex[index]));
      }
      // Fill reconstucted PV info
      std::vector<RecPVInfo> recpvvec;
      std::vector<PV::Vertex*>::iterator itRecV;
      for (itRecV = vecOfVertices.begin(); vecOfVertices.end() != itRecV; itRecV++) {
        PV::Vertex* pv;
        pv = *itRecV;
        RecPVInfo recinfo;
        recinfo.pRECPV = pv;
        recinfo.x = pv->position.x;
        recinfo.y = pv->position.y;
        recinfo.z = pv->position.z;

        double sigx = sqrt(pv->cov00);
        double sigy = sqrt(pv->cov11);
        double sigz = sqrt(pv->cov22);
        PatPV::XYZPoint a3d(sigx, sigy, sigz);
        recinfo.positionSigma = a3d;
        recinfo.nTracks = pv->nTracks;
        double minRD = 99999.;
        double maxRD = -99999.;
        double chi2 = pv->chi2;
        double nDoF = pv->ndof;

        int mother = 0;
        int velo = 0;
        int lg = 0;
        double d0 = 0;
        double mind0 = 99999.0;
        double maxd0 = -99999.0;
        double trackChi2 = 0.0;
        int tr = 0;

        recinfo.minTrackRD = minRD;
        recinfo.maxTrackRD = maxRD;
        recinfo.mother = mother;
        recinfo.chi2 = chi2;
        recinfo.nDoF = nDoF;
        recinfo.d0 = d0;
        recinfo.d0nTr = (double) d0 / (double) tr;
        recinfo.chi2nTr = (double) trackChi2 / (double) tr;
        recinfo.mind0 = mind0;
        recinfo.maxd0 = maxd0;
        recinfo.nVeloTracks = velo;
        recinfo.nLongTracks = lg;
        recinfo.indexMCPVInfo = -1;
        recpvvec.push_back(recinfo);
      }

      // Fill MC PV info

      // do checking of collision type and mother here or in dumping?

      // vector with MCPVinfo
      std::vector<MCPVInfo> mcpvvec;

      for (std::vector<MCVertex>::iterator itMCV = MC_vertices_selected_events[i_event].begin();
           MC_vertices_selected_events[i_event].end() != itMCV;
           itMCV++) {

        MCPVInfo mcprimvert;
        mcprimvert.pMCPV = &(*itMCV);
        // mcprimvert.nRecTracks = 0;
        mcprimvert.nRecTracks = (*itMCV).numberTracks;
        // mcprimvert.nRecTracks = 99;
        mcprimvert.nRecBackTracks = 0;
        mcprimvert.indexRecPVInfo = -1;
        mcprimvert.nCorrectTracks = 0;
        mcprimvert.multClosestMCPV = 0;
        mcprimvert.distToClosestMCPV = 999999.;
        mcprimvert.decayBeauty = 0;
        mcprimvert.decayCharm = 0;

        mcpvvec.push_back(mcprimvert);
      }

      std::vector<MCPVInfo> rblemcpv;
      std::vector<MCPVInfo> not_rble_but_visible;
      std::vector<MCPVInfo> not_rble;
      int nmrc = 0;

      // count not reconstructible MC PVs
      std::vector<MCPVInfo>::iterator itmc;
      for (itmc = mcpvvec.begin(); mcpvvec.end() != itmc; itmc++) {
        rblemcpv.push_back(*itmc);

        if (itmc->nRecTracks < nTracksToBeRecble) {
          nmrc++;
        }
        if (itmc->nRecTracks < nTracksToBeRecble && itmc->nRecTracks > 1) {
          not_rble_but_visible.push_back(*itmc);
        }
        if (itmc->nRecTracks < nTracksToBeRecble && itmc->nRecTracks < 2) {
          not_rble.push_back(*itmc);
        }
      }

      // match by distance
      for (int ipv = 0; ipv < (int) recpvvec.size(); ipv++) {
        match_mc_vertex_by_distance(ipv, recpvvec, rblemcpv);
      };

      // find nr of false PV

      int nFalsePV = 0;
      int nFalsePV_real = 0;
      for (int ipv = 0; ipv < (int) recpvvec.size(); ipv++) {
        int fake = 0;
        double x = recpvvec[ipv].x;
        double y = recpvvec[ipv].y;
        double z = recpvvec[ipv].z;
        double r = std::sqrt(x * x + y * y);
        double errx = recpvvec[ipv].positionSigma.x;
        double erry = recpvvec[ipv].positionSigma.y;
        double errz = recpvvec[ipv].positionSigma.z;
        double errr = std::sqrt(((x * errx) * (x * errx) + (y * erry) * (y * erry)) / (x * x + y * y));
        double minRDTrack = recpvvec[ipv].minTrackRD;
        double maxRDTrack = recpvvec[ipv].maxTrackRD;
        int mother = recpvvec[ipv].mother;
        double velo = recpvvec[ipv].nVeloTracks;
        double lg = recpvvec[ipv].nLongTracks;
        double d0 = recpvvec[ipv].d0;
        double d0nTr = recpvvec[ipv].d0nTr;
        double chi2nTr = recpvvec[ipv].chi2nTr;
        double mind0 = recpvvec[ipv].mind0;
        double maxd0 = recpvvec[ipv].maxd0;
        double chi2 = recpvvec[ipv].chi2;
        double nDoF = recpvvec[ipv].nDoF;

        if (recpvvec[ipv].indexMCPVInfo < 0) {
          nFalsePV++;
          fake = 1;
          bool vis_found = false;
          for (unsigned int imc = 0; imc < not_rble_but_visible.size(); imc++) {
            if (not_rble_but_visible[imc].indexRecPVInfo > -1) continue;
            double dist = fabs(mcpvvec[imc].pMCPV->z - recpvvec[ipv].z);
            if (dist < 5.0 * recpvvec[ipv].positionSigma.z) {
              vis_found = true;
              not_rble_but_visible[imc].indexRecPVInfo = 10;
              break;
            }
          } // imc
          if (!vis_found) nFalsePV_real++;
        }
      }

      // Fill distance to closest recble MC PV and its multiplicity
      std::vector<MCPVInfo>::iterator itmcl;
      for (itmcl = rblemcpv.begin(); rblemcpv.end() != itmcl; itmcl++) {
        std::vector<MCPVInfo>::iterator cmc = closestMCPV(rblemcpv, itmcl);
        double dist = 999999.;
        int mult = 0;
        if (cmc != rblemcpv.end()) {
          double diff_x = cmc->pMCPV->x - itmcl->pMCPV->x;
          double diff_y = cmc->pMCPV->y - itmcl->pMCPV->y;
          double diff_z = cmc->pMCPV->z - itmcl->pMCPV->z;
          double dist = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
          mult = cmc->nRecTracks;
          itmcl->distToClosestMCPV = dist;
          itmcl->multClosestMCPV = mult;
        }
      }

      // count non.reconstructible close and isolated PVs
      int nmrc_isol = 0;
      int nmrc_close = 0;

      // Counters
      int nMCPV = rblemcpv.size() - nmrc;
      int nRecMCPV = 0;
      int nMCPV_isol = 0;
      int nRecMCPV_isol = 0;
      int nMCPV_close = 0;
      int nRecMCPV_close = 0;

      for (itmc = rblemcpv.begin(); rblemcpv.end() != itmc; itmc++) {
        if (itmc->distToClosestMCPV > dzIsolated) nMCPV_isol++;
        if (itmc->distToClosestMCPV > dzIsolated && itmc->nRecTracks < nTracksToBeRecble) nmrc_isol++;
        if (itmc->distToClosestMCPV < dzIsolated) nMCPV_close++;
        if (itmc->distToClosestMCPV < dzIsolated && itmc->nRecTracks < nTracksToBeRecble) nmrc_close++;
        if (itmc->indexRecPVInfo > -1) {
          nRecMCPV++;
          if (itmc->distToClosestMCPV > dzIsolated) nRecMCPV_isol++;
          if (itmc->distToClosestMCPV < dzIsolated) nRecMCPV_close++;
        }
      }

      nMCPV_isol = nMCPV_isol - nmrc_isol;
      nMCPV_close = nMCPV_close - nmrc_close;

      sum_nMCPV += nMCPV;
      sum_nRecMCPV += nRecMCPV;
      sum_nMCPV_isol += nMCPV_isol;
      sum_nRecMCPV_isol += nRecMCPV_isol;
      sum_nMCPV_close += nMCPV_close;
      sum_nRecMCPV_close += nRecMCPV_close;
      sum_nFalsePV += nFalsePV;
      sum_nFalsePV_real += nFalsePV_real;

      // loop over matched MC PVs and get pull and errors
      for (auto mc_vertex_info : rblemcpv) {
        int rec_index = mc_vertex_info.indexRecPVInfo;
        if (rec_index < 0) continue;
        MCVertex* mc_vertex = mc_vertex_info.pMCPV;
        double diff_x = recpvvec[rec_index].x - mc_vertex->x;
        double diff_y = recpvvec[rec_index].y - mc_vertex->y;
        double diff_z = recpvvec[rec_index].z - mc_vertex->z;
        vec_diff_x.push_back(diff_x);
        vec_diff_y.push_back(diff_y);
        vec_diff_z.push_back(diff_z);

        double err_x = recpvvec[rec_index].positionSigma.x;
        double err_y = recpvvec[rec_index].positionSigma.y;
        double err_z = recpvvec[rec_index].positionSigma.z;

        vec_err_x.push_back(err_x);
        vec_err_y.push_back(err_y);
        vec_err_z.push_back(err_z);
      }
    } // end loop over files/events

    info_cout.precision(4);
    info_cout << " ============================================" << std::endl;
    info_cout << " Efficiencies for reconstructible MC vertices: " << std::endl;
    info_cout << " ============================================" << std::endl;
    info_cout << " " << std::endl;

    info_cout << " MC PV is reconstructible if at least " << nTracksToBeRecble << "  tracks are reconstructed"
              << std::endl;
    info_cout << " MC PV is isolated if dz to closest reconstructible MC PV >  " << dzIsolated << " mm" << std::endl;
    std::string ff = "by counting tracks";
    if (!matchByTracks) ff = "by dz distance";
    info_cout << " REC and MC vertices matched:  " << ff << std::endl;

    info_cout << " " << std::endl;

    printRat("All", sum_nRecMCPV, sum_nMCPV);
    printRat("Isolated", sum_nRecMCPV_isol, sum_nMCPV_isol);
    printRat("Close", sum_nRecMCPV_close, sum_nMCPV_close);
    printRat("False rate", sum_nFalsePV, sum_nRecMCPV + sum_nFalsePV);
    printRat("Real false rate", sum_nFalsePV_real, sum_nRecMCPV + sum_nFalsePV_real);

    info_cout << "new found: " << sum_nRecMCPV << " / " << sum_nMCPV << std::endl;
    info_cout << "new fakes: " << sum_nFalsePV << std::endl << std::endl;

// save information about matched reconstructed PVs for pulls distributions
#ifdef WITH_ROOT
    TFile* out_fille = new TFile(("../output/" + mode + "_PVChecker.root").data(), "RECREATE");
    TTree* tree = new TTree("PV_tree", "PV_tree");
    // double x_true, y_true, z_true;
    double diff_x, diff_y, diff_z;
    double err_x, err_y, err_z;

    tree->Branch("diff_x", &diff_x);
    tree->Branch("diff_y", &diff_y);
    tree->Branch("diff_z", &diff_z);

    tree->Branch("err_x", &err_x);
    tree->Branch("err_y", &err_y);
    tree->Branch("err_z", &err_z);

    // tree->Branch("x_true", &x_true);
    // tree->Branch("y_true", &y_true);
    // tree->Branch("z_true", &z_true);

    for (int i = 0; i < vec_diff_x.size(); i++) {
      diff_x = vec_diff_x.at(i);
      diff_y = vec_diff_y.at(i);
      diff_z = vec_diff_z.at(i);

      err_x = vec_err_x.at(i);
      err_y = vec_err_y.at(i);
      err_z = vec_err_z.at(i);

      tree->Fill();
    }
    tree->Write();
    out_fille->Close();
#endif
  }
}
