#include "../include/run_PatPV_CPU.h"


/*
XYZPoint& seedPoint,
              std::vector<Track*>& rTracks,
             Vertex& vtx,
             std::vector<Track*>& tracks2remove

*/





bool reconstructMultiPVFromTracks( VeloState * tracks2use,
                                                       Vertex * outvtxvec, int host_number_of_tracks_pinned,
  uint * number_of_vertex, int event_number) 
{
  

  VeloState * rtracks = tracks2use;

  //outvtxvec.clear();


  PVSeedTool seedtool;
  //double m_beamSpotX = 0.02;
  //double m_beamSpotY = -0.16;
  double m_beamSpotX = 0.;
  double m_beamSpotY = 0.;
  XYZPoint beamspot{m_beamSpotX, m_beamSpotY, 0.};
  

    


  int nvtx_before = -1;
  int nvtx_after  =  0;
  //for (int i = 0; i < 5 ; i++) {
  //do we really need this loop?
  //while ( nvtx_after > nvtx_before ) {
    nvtx_before = nvtx_after;
    // reconstruct vertices


  AdaptivePV3DFitter fitter;
  std::vector<XYZPoint> seeds = seedtool.getSeeds(rtracks, beamspot, host_number_of_tracks_pinned);
    for ( auto seed : seeds) {
      Vertex recvtx;


      //VeloState * tracks2remove;
      std::vector<VeloState> tracks2remove;
      // fitting
      bool scvfit = fitter.fitVertex( seed, rtracks, recvtx, tracks2remove, host_number_of_tracks_pinned);
      if (!scvfit) continue;
      
      

      
      outvtxvec[event_number *max_number_vertices + nvtx_after] = recvtx;
      nvtx_after++;
      //removeTracks(rtracks, tracks2remove);
    }//iterate on seeds
    number_of_vertex[event_number] = nvtx_after;
  //}//iterate on vtx

  return true;

}




int run_PatPV_on_CPU (
  VeloState * host_velo_states,
  int * host_accumulated_tracks,
  uint* host_velo_track_hit_number_pinned,
  VeloTracking::Hit<true>* host_velo_track_hits_pinned,
  int * host_number_of_tracks_pinned,
  const int &number_of_events,
  Vertex * outvtxvec,
  uint * number_of_vertex
) {

XYZPoint beamspot(0.,0.,0.);
PVSeedTool seedtool;
//std:std::vector<XYZPoint> seeds = seedtool.getSeeds(host_velo_states, beamspot, *host_number_of_tracks_pinned);

/*
AdaptivePV3DFitter fitter;
Vertex recvtx;
std::vector<VeloState> tracks2remove;
XYZPoint seed = seeds.at(0);
fitter.fitVertex(seed, host_velo_states, recvtx, tracks2remove, number_of_events);
*/
//Vertex  outvtxvec[100];
//std::vector<VeloState> velostate_vec;

//for(int i = 0; i < *host_number_of_tracks_pinned; i++)  velostate_vec.push_back(host_velo_states[i]); 


for(int i_event = 0; i_event < number_of_events; i_event++) {

  int number_of_tracks = host_number_of_tracks_pinned[i_event];
 VeloState * state_base_pointer = host_velo_states + 2 * host_accumulated_tracks[i_event];
VeloState  kalman_states[number_of_tracks];

//recovers previusoly found vertices in first event
for(int i = 0; i < number_of_tracks; i++) kalman_states[i] = state_base_pointer[2*i ];
std::cout << "least: " << kalman_states[1].x <<std::endl;
std::cout << "least: " << kalman_states[1].y <<std::endl;
std::cout << "least: " << kalman_states[1].z <<std::endl;
std::cout << "least: " << kalman_states[1].tx <<std::endl;
std::cout << "least: " << kalman_states[1].ty <<std::endl;
std::cout << "least: " << kalman_states[1].c00 <<std::endl;
std::cout << "least: " << kalman_states[1].c20 <<std::endl;
std::cout << "least: " << kalman_states[1].c22 <<std::endl;
std::cout << "least: " << kalman_states[1].c11 <<std::endl;
std::cout << "least: " << kalman_states[1].c31 <<std::endl;
std::cout << "least: " << kalman_states[1].c33 <<std::endl;

  //not workign yet
for(int i = 0; i < number_of_tracks; i++) kalman_states[i] = state_base_pointer[2*i +1];
std::cout << "kalman: " << kalman_states[1].x <<std::endl;
std::cout << "kalman: " << kalman_states[1].y <<std::endl;
std::cout << "kalman: " << kalman_states[1].z <<std::endl;
std::cout << "kalman: " << kalman_states[1].tx <<std::endl;
std::cout << "kalman: " << kalman_states[1].ty <<std::endl;
std::cout << "kalman: " << kalman_states[1].c00 <<std::endl;
std::cout << "kalman: " << kalman_states[1].c20 <<std::endl;
std::cout << "kalman: " << kalman_states[1].c22 <<std::endl;
std::cout << "kalman: " << kalman_states[1].c11 <<std::endl;
std::cout << "kalman: " << kalman_states[1].c31 <<std::endl;
std::cout << "least: " << kalman_states[1].c33 <<std::endl;

reconstructMultiPVFromTracks(kalman_states, outvtxvec, host_number_of_tracks_pinned[i_event], number_of_vertex, i_event);
}

  return 0;
}


struct MCVertex {
  double x;
  double y;
  double z;
  int numberTracks;
};


void checkPVs(  const std::string& foldername,  const bool& fromNtuple, uint number_of_files)


{
   std::cout << "Checking PVs: " << std::endl;
   std::vector<std::string> folderContents = list_folder(foldername, fromNtuple);
  
  uint requestedFiles = number_of_files==0 ? folderContents.size() : number_of_files;
  verbose_cout << "Requested " << requestedFiles << " files" << std::endl;

  if ( requestedFiles > folderContents.size() ) {
    error_cout << "ERROR: requested " << requestedFiles << " files, but only " << folderContents.size() << " files are present" << std::endl;
    exit(-1);
  }

  int readFiles = 0;
  
  //vector containing for each event vector of MCVertices
  std::vector<std::vector<MCVertex>> events_vertices;
  
  for (uint i=0; i<requestedFiles; ++i) {
    // Read event #i in the list and add it to the inputs
    // if more files are requested than present in folder, read them again
    std::string readingFile = folderContents[i % folderContents.size()];
    std::string filename = foldername + "/" + readingFile;

    // Check if file exists
    if (!fileExists(filename)){
    throw StrException("Error: File " + filename + " does not exist.");
     }
   
    TFile *file = new TFile(filename.data(),"READ");
    TTree *tree = (TTree*)file->Get("Hits_detectors");
    assert( tree);

    TBranch        *b_ovtx_x;   //!
    TBranch        *b_ovtx_y;   //!
    TBranch        *b_ovtx_z;   //!
    TBranch        *b_p;
    TBranch        *b_MCVertexType;
    TBranch        *b_VertexNumberOfTracks;
    TBranch        *b_MCVertexNumberOfTracks;


    Double_t        ovtx_x;
    Double_t        ovtx_y;
    Double_t        ovtx_z;
    Double_t        p;
    Int_t           MCVertexType;
    Int_t           VertexNumberOfTracks;
    Int_t           MCVertexNumberOfTracks;
    

    TTree* fChain = tree;

    fChain->SetBranchAddress("ovtx_x", &ovtx_x, &b_ovtx_x);
    fChain->SetBranchAddress("ovtx_y", &ovtx_y, &b_ovtx_y);
    fChain->SetBranchAddress("ovtx_z", &ovtx_z, &b_ovtx_z);
    fChain->SetBranchAddress("p", &p, &b_p);
    fChain->SetBranchAddress("MCVertexType", &MCVertexType, &b_MCVertexType);
    fChain->SetBranchAddress("VertexNumberOfTracks", &VertexNumberOfTracks, &b_VertexNumberOfTracks);
    fChain->SetBranchAddress("MCVertexNumberOfTracks", &MCVertexNumberOfTracks, &b_MCVertexNumberOfTracks);

    // vector containing MC vertices
    std::vector<MCVertex> vertices;

    std::vector<double> found_z;
    Long64_t maxEntries = fChain->GetTree()->GetEntries();
    for(Long64_t entry = 0; entry< maxEntries ; ++entry){
      fChain->GetTree()->GetEntry(entry);
      if( p<0 || MCVertexType != 1) continue;  // Hits not associated to an MCP are stored with p < 0 and only look for trakcs from PVs

      if(std::find(found_z.begin(), found_z.end(), ovtx_z) == found_z.end()) {
        std::cout << "found a new vertex with: " << std::endl;
        std::cout << "x: " << ovtx_x << std::endl;
        std::cout << "y: " << ovtx_y << std::endl;
        std::cout << "z: " << ovtx_z << std::endl;
        std::cout << "number of tracks: " << MCVertexNumberOfTracks << std::endl;
        found_z.push_back(ovtx_z);
        MCVertex vertex;
        vertex.x = ovtx_x;
        vertex.y = ovtx_y;
        vertex.z = ovtx_z;
        vertex.numberTracks = MCVertexNumberOfTracks;
        vertices.push_back(vertex);
      }

    }

    events_vertices.push_back(vertices);
  }
  std::cout << "here are the vertices!" << std::endl;
  for(auto vtx_vec : events_vertices) {
    for (auto vtx : vtx_vec) {
      std::cout << "x: " << vtx.x << std::endl;
        std::cout << "y: " << vtx.y << std::endl;
        std::cout << "z: " << vtx.z << std::endl;
        std::cout << "number of tracks: " << vtx.numberTracks << std::endl;
    }
  }


  //now, for each event, loop over MCVertices and check if it has been found (distance criterion) -> calculate efficiency


}
