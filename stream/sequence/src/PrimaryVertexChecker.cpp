#include "PrimaryVertexChecker.h"
#include "../include/run_PatPV_CPU.h"
//#include "../../../PatPV/include/PVSeedTool.h"



void checkPVs(  const std::string& foldername,  const bool& fromNtuple, uint number_of_files, Vertex * rec_vertex, uint* number_of_vertex)


{
   std::cout << "Checking PVs: " << std::endl;
   std::vector<std::string> folderContents = list_folder(foldername);
  
  uint requestedFiles = number_of_files==0 ? folderContents.size() : number_of_files;
  verbose_cout << "Requested " << requestedFiles << " files" << std::endl;

  if ( requestedFiles > folderContents.size() ) {
    error_cout << "ERROR: requested " << requestedFiles << " files, but only " << folderContents.size() << " files are present" << std::endl;
    exit(-1);
  }

  int readFiles = 0;
  
  //vector containing for each event vector of MCVertices
  std::vector<std::vector<MCVertex>> events_vertices;

      //counters for efficiency
    int number_reconstructible_vertices = 0; 
    int number_reconstructed_vertices = 0;
    int number_fake_vertices = 0;


// vector containing MC vertices
    //std::vector<MCVertex> vertices;

 //counters for efficiencies/fake rate
  int m_nMCPV = 0;
  int m_nRecMCPV  = 0;
  int m_nMCPV_isol = 0;
  int m_nRecMCPV_isol = 0;
  int m_nMCPV_close = 0;
  int m_nRecMCPV_close = 0;
  int m_nFalsePV = 0;
  int m_nFalsePV_real = 0;
  
  //loop over files/events
  for (uint i_event=0; i_event<requestedFiles; ++i_event) {
    // Read event #i in the list and add it to the inputs
    // if more files are requested than present in folder, read them again

    //collect true PV vertices in a event
    std::string readingFile = folderContents[i_event % folderContents.size()];
    std::string filename = foldername + "/" + readingFile;
    std::vector<char> inputContents;
    readFileIntoVector(foldername + "/" + readingFile, inputContents);

    uint8_t* input = (uint8_t*) inputContents.data();

    int number_mcpv = *((int*)  input); input += sizeof(int);
    //std::cout << "num MCPs = " << number_mcp << std::endl;
    std::vector<MCVertex> MC_vertices;
    for (uint32_t i=0; i<number_mcpv; ++i) {
      MCVertex mc_vertex;

      int VertexNumberOfTracks = *((int*)  input); input += sizeof(int);
      if(VertexNumberOfTracks >= 4) number_reconstructible_vertices++;
      mc_vertex.numberTracks = VertexNumberOfTracks;
      mc_vertex.x = *((double*)  input); input += sizeof(double);
      mc_vertex.y = *((double*)  input); input += sizeof(double);
      mc_vertex.z = *((double*)  input); input += sizeof(double);
      std::cout << "read MC vertex " << i << std::endl;
      std::cout << "nubmer tracks: " << mc_vertex.numberTracks << std::endl;
      std::cout << "x: " << mc_vertex.x << std::endl;
      std::cout << "y: " << mc_vertex.y << std::endl;
      std::cout << "z: " << mc_vertex.z << std::endl;

      //if(mc_vertex.numberTracks >= 4) vertices.push_back(mc_vertex);
      MC_vertices.push_back(mc_vertex);
    }
    

//fill a vector with bools to check for fakes
    std::vector<bool> isFake;
    for(uint i = 0; i < number_of_vertex[i_event]; i++) isFake.push_back(true);

for (auto vtx : MC_vertices) {
          if(vtx.numberTracks < 4) continue;
    //collect reconstruced vertices in a event
          bool matched = false;
    for(uint i = 0; i < number_of_vertex[i_event]; i++) {
      int index = i_event  * PatPV::max_number_vertices + i;
      double r2 = rec_vertex[index].x*rec_vertex[index].x + rec_vertex[index].y * rec_vertex[index].y;
      //radial cut against fake vertices
      double r = 0.;
      if(rec_vertex[index].tracks.size() < 10) r = 0.2;
      else r = 0.4;
      if(r2 >  r*r) continue;



      
        
        //for each reconstructed PV, loop over MC PVs
        
        
          //number_reconstructible_vertices++;
          
          //don't forget that covariance is sigma squared!
          if(abs(rec_vertex[index].z - vtx.z) <  5. * sqrt(rec_vertex[index].cov22)) {

            number_reconstructed_vertices++;
            matched = true;
            isFake.at(i) = false;
            break;
          }

        }
        //if(!matched) number_fake_vertices++; 

     }


     for (auto fake : isFake) {if (fake) number_fake_vertices++;};


    //events_vertices.push_back(vertices);


  //try to implement nominal PV checker

  std::vector<Vertex*> vecOfVertices;
  //first fill vector with vertices
  for(uint i = 0; i < number_of_vertex[i_event]; i++) {
    int index = i_event  * PatPV::max_number_vertices + i;
    vecOfVertices.push_back(&(rec_vertex[index]));
  }
  // Fill reconstucted PV info
  std::vector<RecPVInfo> recpvvec;
  std::vector<Vertex*>::iterator itRecV;
  for(itRecV = vecOfVertices.begin(); vecOfVertices.end() != itRecV;
               itRecV++) {
    Vertex* pv;
    pv = *itRecV;
    RecPVInfo recinfo;
    recinfo.pRECPV= pv;
    recinfo.x = pv->x;
    recinfo.y = pv->y;
    recinfo.z = pv->z;

    
    double sigx = sqrt(pv->cov00);
    double sigy = sqrt(pv->cov11);
    double sigz = sqrt(pv->cov22);
    XYZPoint a3d(sigx,sigy,sigz);
    recinfo.positionSigma = a3d;
    recinfo.nTracks = pv->tracks.size();
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
    recinfo.d0nTr = (double)d0/(double)tr;
    recinfo.chi2nTr = (double)trackChi2/(double)tr; 
    recinfo.mind0 = mind0;
    recinfo.maxd0 = maxd0;
    recinfo.nVeloTracks = velo; 
    recinfo.nLongTracks = lg; 
    recinfo.indexMCPVInfo = -1;
    recpvvec.push_back(recinfo);
    
  }

    // Fill MC PV info
  

  //do checking of collision type and mother here or in dumping?
  
  
  //vecotr with MCPVinfo
 std::vector<MCPVInfo> mcpvvec;
                   
  for(std::vector<MCVertex>::iterator itMCV = MC_vertices.begin();
      MC_vertices.end() != itMCV; itMCV++) {
    
    
      
        MCPVInfo mcprimvert;
        mcprimvert.pMCPV = &(*itMCV);
        //mcprimvert.nRecTracks = 0;
        mcprimvert.nRecTracks = (*itMCV).numberTracks;
        //mcprimvert.nRecTracks = 99;
        mcprimvert.nRecBackTracks = 0;
        mcprimvert.indexRecPVInfo = -1;
        mcprimvert.nCorrectTracks = 0;
        mcprimvert.multClosestMCPV = 0;
        mcprimvert.distToClosestMCPV = 999999.;
        mcprimvert.decayBeauty = 0;
        mcprimvert.decayCharm  = 0;
        
        mcpvvec.push_back(mcprimvert);
      
    
  }
  std::cout << "nubmer of MCPVinfo:" << i_event << " " << mcpvvec.size() << std::endl;

  std::vector<MCPVInfo> rblemcpv;
  std::vector<MCPVInfo> not_rble_but_visible;
  std::vector<MCPVInfo> not_rble;
  int nmrc = 0;

  //configuration for PV checker -> check values
int m_nTracksToBeRecble = 5;
double m_dzIsolated = 10;

  std::vector<MCPVInfo>::iterator itmc;
  for (itmc = mcpvvec.begin(); mcpvvec.end() != itmc; itmc++) {
    rblemcpv.push_back(*itmc);
    std::cout << "number of tracks: " << (*itmc).nRecTracks << std::endl;;
    if (itmc->nRecTracks < m_nTracksToBeRecble)
      {
        nmrc++;
      }
    if(itmc->nRecTracks < m_nTracksToBeRecble && itmc->nRecTracks > 1)
      {
        not_rble_but_visible.push_back(*itmc);
      }
    if(itmc->nRecTracks < m_nTracksToBeRecble && itmc->nRecTracks < 2)
      {
        not_rble.push_back(*itmc);  
      }

  }
  std::cout << "nubmer of rblemcpv:" << i_event << " " << rblemcpv.size() << std::endl;
  std::cout << "nubmer of nmrc:" << i_event << " " << nmrc << std::endl;

      for(int ipv = 0; ipv < (int) recpvvec.size(); ipv++) {
      match_mc_vertex_by_distance(ipv, recpvvec, rblemcpv);
    };


    // find nr of false PV

  int nFalsePV = 0;
  int nFalsePV_real = 0;
  for(int ipv = 0; ipv < (int) recpvvec.size(); ipv++) {
    int fake = 0;
    double x = recpvvec[ipv].x;
    double y = recpvvec[ipv].y;
    double z = recpvvec[ipv].z;
    double r = std::sqrt(x*x + y*y);
    double errx = recpvvec[ipv].positionSigma.x;
    double erry = recpvvec[ipv].positionSigma.y;
    double errz = recpvvec[ipv].positionSigma.z;
    double errr = std::sqrt(((x*errx)*(x*errx)+(y*erry)*(y*erry))/(x*x+y*y));
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


    if (recpvvec[ipv].indexMCPVInfo < 0 ) {
      nFalsePV++; 
      fake = 1;
      bool vis_found = false;
      for(unsigned int imc = 0; imc < not_rble_but_visible.size() ; imc++) {
        if ( not_rble_but_visible[imc].indexRecPVInfo > -1 ) continue;
        double dist = fabs(mcpvvec[imc].pMCPV->z - recpvvec[ipv].z);
        if (  dist < 5.0 * recpvvec[ipv].positionSigma.z ) {
          vis_found = true;
    not_rble_but_visible[imc].indexRecPVInfo = 10;
          break;
  }
      } // imc
      if ( !vis_found ) nFalsePV_real++;
    }
 }


   // Counters
  int nMCPV                 = rblemcpv.size()-nmrc;
  int nRecMCPV              = 0;
  int nMCPV_isol            = 0;
  int nRecMCPV_isol         = 0;
  int nMCPV_close           = 0;
  int nRecMCPV_close        = 0;



  for(itmc = rblemcpv.begin(); rblemcpv.end() != itmc; itmc++) {
    if(itmc->distToClosestMCPV > m_dzIsolated) nMCPV_isol++;
    if(itmc->distToClosestMCPV < m_dzIsolated) nMCPV_close++;
    if(itmc->indexRecPVInfo > -1) {
      nRecMCPV++;
      if(itmc->distToClosestMCPV > m_dzIsolated) nRecMCPV_isol++;
      if(itmc->distToClosestMCPV < m_dzIsolated) nRecMCPV_close++;
    }
  }

    m_nMCPV                 +=  nMCPV;
  m_nRecMCPV              +=  nRecMCPV;
  m_nMCPV_isol            +=  nMCPV_isol;
  m_nRecMCPV_isol         +=  nRecMCPV_isol;
  m_nMCPV_close           +=  nMCPV_close;
  m_nRecMCPV_close        +=  nRecMCPV_close;
  m_nFalsePV              +=  nFalsePV;
  m_nFalsePV_real         +=  nFalsePV_real;


  MC_vertices.clear();
  } //end loop over files/events
  

  std::cout << "found " << number_reconstructed_vertices << " / " << number_reconstructible_vertices << " vertices! -> efficiency: " << (double)number_reconstructed_vertices / (double)number_reconstructible_vertices << std::endl; 
  std::cout << "fakes: " << number_fake_vertices << std::endl;



            printRat("All",       m_nRecMCPV ,       m_nMCPV );
            printRat("Isolated",  m_nRecMCPV_isol,   m_nMCPV_isol );
            printRat("Close",     m_nRecMCPV_close,  m_nMCPV_close );
            printRat("False rate",m_nFalsePV ,       m_nRecMCPV+m_nFalsePV );


          printRat("Real false rate", m_nFalsePV_real , m_nRecMCPV+m_nFalsePV_real);

   std::cout <<   "new found: " <<     m_nRecMCPV << " / " << m_nMCPV  << std::endl;
   std::cout << "new fakes: " << m_nFalsePV << std::endl;

          
    
    
    
 

}
