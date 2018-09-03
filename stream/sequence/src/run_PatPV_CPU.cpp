#include "../include/run_PatPV_CPU.h"
//#include "../../../PatPV/include/PVSeedTool.h"
#include "../../../PatPV/include/PVSeedTool.h"




bool reconstructMultiPVFromTracks(VeloState * tracks2use, Vertex * outvtxvec, int host_number_of_tracks_pinned,
  uint * number_of_vertex, int event_number, bool * tracks2disable, XYZPoint * seeds, uint * number_of_seeds) 
{
  

  VeloState * rtracks = tracks2use;
 
  

    
  //PatPv::max_number_vertices


  int nvtx_after  =  0;

    // reconstruct vertices


  
  int number_rec_vtx = 0;
  bool continue_fitting = true;
  while(continue_fitting) {
    XYZPoint beamspot = {0.,0.,0.};
    getSeeds( rtracks, beamspot, host_number_of_tracks_pinned,  seeds, number_of_seeds,  event_number, tracks2disable);
    int before_fit = nvtx_after;
    for(int i=0; i < number_of_seeds[event_number]; i++) {
      XYZPoint seed = seeds[event_number * PatPV::max_number_vertices + i ]; 
      Vertex recvtx;

      std::cout << "trying to fit with seed " << i << std::endl;
      std::cout << seed.x << std::endl;
      std::cout << seed.y << std::endl;
      std::cout << seed.z << std::endl;



      // fitting
      bool scvfit = fitVertex( seed, rtracks, recvtx, host_number_of_tracks_pinned, tracks2disable);
      if (!scvfit) continue;
      std::cout<<"got vertex " << std::endl; 
      std::cout << recvtx.x << std::endl;
      std::cout << recvtx.y << std::endl;
      std::cout << recvtx.z << std::endl;
      
      

      
      outvtxvec[event_number * PatPV::max_number_vertices + nvtx_after] = recvtx;
      nvtx_after++;

    }//iterate on seeds
    if(before_fit == nvtx_after) continue_fitting = false;
    
    
  }
    number_of_vertex[event_number] = nvtx_after;


  return true;
  

}




int run_PatPV_on_CPU (
  VeloState * host_velo_states,
  int * host_accumulated_tracks,
  int * host_number_of_tracks_pinned,
  const int &number_of_events,
  Vertex * outvtxvec,
  uint * number_of_vertex,
  XYZPoint * seeds,
  uint * number_of_seeds
) {

XYZPoint beamspot(0.,0.,0.);






for(int i_event = 0; i_event < number_of_events; i_event++) {

  int number_of_tracks = host_number_of_tracks_pinned[i_event];
 VeloState * state_base_pointer = host_velo_states + 2 * host_accumulated_tracks[i_event];
VeloState  kalman_states[number_of_tracks];


 bool  tracks2disable[number_of_tracks];

  //works
for(int i = 0; i < number_of_tracks; i++) kalman_states[i] = state_base_pointer[2*i +1];
for(int i = 0; i < number_of_tracks; i++) tracks2disable[i] = false;

XYZPoint beamspot = {0.,0.,0.};
//getSeeds( kalman_states, beamspot, number_of_tracks,  seeds, number_of_seeds,  i_event, tracks2disable);
reconstructMultiPVFromTracks(kalman_states, outvtxvec, host_number_of_tracks_pinned[i_event], number_of_vertex, i_event, tracks2disable, seeds, number_of_seeds);
}




  return 0;
}


struct MCVertex {
  double x;
  double y;
  double z;
  int numberTracks;
};


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
  
  for (uint i=0; i<requestedFiles; ++i) {
    // Read event #i in the list and add it to the inputs
    // if more files are requested than present in folder, read them again
    std::string readingFile = folderContents[i % folderContents.size()];
    std::string filename = foldername + "/" + readingFile;


   
    TFile *file = new TFile(filename.data(),"READ");
    TTree *tree = (TTree*)file->Get("pv");
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
    Int_t           VertexNumberOfTracks;
    Int_t           MCVertexNumberOfTracks;
    

    TTree* fChain = tree;

    fChain->SetBranchAddress("pv_x", &ovtx_x, &b_ovtx_x);
    fChain->SetBranchAddress("pv_y", &ovtx_y, &b_ovtx_y);
    fChain->SetBranchAddress("pv_z", &ovtx_z, &b_ovtx_z);

    fChain->SetBranchAddress("number_rec_tracks", &VertexNumberOfTracks, &b_VertexNumberOfTracks);
   //fChain->SetBranchAddress("MCVertexNumberOfTracks", &MCVertexNumberOfTracks, &b_MCVertexNumberOfTracks);

    // vector containing MC vertices
    std::vector<MCVertex> vertices;



    
    Long64_t maxEntries = fChain->GetTree()->GetEntries();
    std::cout << "-------" << std::endl;
    for(Long64_t entry = 0; entry< maxEntries ; ++entry){
      fChain->GetTree()->GetEntry(entry);


        std::cout <<  "MC vertex " << entry << std::endl;
        std::cout << "x: " << ovtx_x << std::endl;
        std::cout << "y: " << ovtx_y << std::endl;
        std::cout << "z: " << ovtx_z << std::endl;
        std::cout << "number of reconstructible tracks: " << VertexNumberOfTracks << std::endl;
        if(VertexNumberOfTracks >= 4) number_reconstructible_vertices++;
        
        MCVertex vertex;
        vertex.x = ovtx_x;
        vertex.y = ovtx_y;
        vertex.z = ovtx_z;
        vertex.numberTracks = VertexNumberOfTracks;
        vertices.push_back(vertex);
      

    }
    std::cout << " ----" << std::endl;

    events_vertices.push_back(vertices);
  }
  std::cout << "here are the MC vertices!" << std::endl;
  for(auto vtx_vec : events_vertices) {
    for (auto vtx : vtx_vec) {
      std::cout << "x: " << vtx.x << std::endl;
        std::cout << "y: " << vtx.y << std::endl;
        std::cout << "z: " << vtx.z << std::endl;
        std::cout << "number of tracks: " << vtx.numberTracks << std::endl;
    }
  }

  std::cout << "------------" << std::endl;
  std::cout << "rec vertices with errors:" << std::endl;
  for(int i = 0; i < number_of_vertex[0]; i++) {
    int index = 0  * PatPV::max_number_vertices + i;
    std::cout << std::setprecision(4) << "x: " << rec_vertex[index].x << " " << rec_vertex[index].cov00 << std::endl;
    std::cout << std::setprecision(4) << "y: " << rec_vertex[index].y << " " << rec_vertex[index].cov11 << std::endl;
    std::cout << std::setprecision(4) << "z: " << rec_vertex[index].z << " " << rec_vertex[index].cov22 << std::endl;
  }


  //now, for each event, loop over MCVertices and check if it has been found (distance criterion) -> calculate efficiency
  //again, loop over events/files
  //loop first over rec vertices, hten over files/events
  /*
  for(int i_event = 0; i_event < number_of_files; i_event++) {
    std::cout << "number_of_vertex: " << number_of_vertex[i_event] << std::endl;
    for(uint i = 0; i < number_of_vertex[i_event]; i++) {
      int index = i_event  *max_number_vertices + i;
      std::cout << "reconstructed vertex " << i << std::endl;
        std::cout << rec_vertex[index].pos.x << std::endl;
        std::cout << rec_vertex[index].pos.y << std::endl;
        std::cout << rec_vertex[index].pos.z << std::endl;
    }
  }
*/

  int i_event = 0;


  std::ofstream pull_file;
  pull_file.open ("../pulls.txt");

  //loop over events/files
  std::cout << "start comparison" << std::endl;
  for(int i_event = 0; i_event < number_of_files; i_event++) {
    std::cout << "-------------- event " << i_event << "____" << std::endl;
    //for each file, loop over reconstructed PVs
    for(uint i = 0; i < number_of_vertex[i_event]; i++) {
      int index = i_event  * PatPV::max_number_vertices + i;
      double r2 = rec_vertex[index].x*rec_vertex[index].x + rec_vertex[index].y * rec_vertex[index].y;
      //radial cut against fake vertices
      double r = 0.;
      if(rec_vertex[index].tracks.size() < 10) r = 0.2;
      else r = 0.4;
      if(r2 >  r*r) continue;
      

      //double r2 = rec_vertex[index].pos.x * rec_vertex[index].pos.x + rec_vertex[index].pos.y * rec_vertex[index].pos.y;
      //try r cut to reduce fake rate
      //if(r2 > 0.02) continue;
      //look if reconstructed vertex matches with reconstrutible by distance criterion
        
      
        bool matched = false;
        //for each reconstructed PV, loop over MC PVs
        auto vtx_vec = events_vertices.at(i_event);
        for (auto vtx : vtx_vec) {
          if(vtx.numberTracks < 4) continue;
          //number_reconstructible_vertices++;
          
          //don't forget that covariance is sigma squared!
          if(abs(rec_vertex[index].z - vtx.z) <  5. * sqrt(rec_vertex[index].cov22)) {

            std::cout << "matched a vertex" << std::endl;
            std::cout << std::setprecision(4) <<"x: " << rec_vertex[index].x << " " << vtx.x << " " <<  rec_vertex[index].cov00 << std::endl;
            std::cout << std::setprecision(4) <<"y: " << rec_vertex[index].y << " " << vtx.y << " " <<  rec_vertex[index].cov11 << std::endl;
            std::cout << std::setprecision(4) <<"z: " << rec_vertex[index].z << " " << vtx.z << " " <<  rec_vertex[index].cov22 << std::endl;
            pull_file << rec_vertex[index].x << " " << rec_vertex[index].y << " " << rec_vertex[index].z << " " << vtx.x << " " << vtx.y << " " << vtx.z << " " << sqrt(rec_vertex[index].cov00) << " " << sqrt(rec_vertex[index].cov11) << " " << sqrt(rec_vertex[index].cov22) << "\n";
            number_reconstructed_vertices++;
            matched = true;
            break;
          }
        }
        
        if(!matched) {number_fake_vertices++; 
       std::cout << "have a fake vertex: " << std::endl;
       std::cout << std::setprecision(4) <<"x: " << rec_vertex[index].x << " " <<  rec_vertex[index].cov00 << std::endl;
            std::cout << std::setprecision(4) <<"y: " << rec_vertex[index].y << " " <<  rec_vertex[index].cov11 << std::endl;
            std::cout << std::setprecision(4) <<"z: " << rec_vertex[index].z << " "  <<  rec_vertex[index].cov22 << std::endl;
        }
      }
    
  }
  std::cout << "end comparison" << std::endl;

  std::cout << "found " << number_reconstructed_vertices << " / " << number_reconstructible_vertices << " vertices! -> efficiency: " << (double)number_reconstructed_vertices / (double)number_reconstructible_vertices << std::endl; 
  std::cout << "fakes: " << number_fake_vertices << std::endl;
    
    
    
  


/*
  int i_event = 0;
  for(auto vtx_vec : events_vertices) {
    for (auto vtx : vtx_vec) {
      //std::cout << "x: " << vtx.x << std::endl;
        //std::cout << "y: " << vtx.y << std::endl;
        //std::cout << "z: " << vtx.z << std::endl;
        //std::cout << "number of tracks: " << vtx.numberTracks << std::endl;
      //only look at MC vertices with enough tracks
      if(vtx.numberTracks < 4) continue;
      number_reconstructible_vertices++;
      std::cout << "-----------" << std::endl; 
      bool 
      for(uint i = 0; i < number_of_vertex[i_event]; i++) {
        int index = i_event  *max_number_vertices + i;
        //std::cout << std::setprecision(4) << "vertex " << i << " " << rec_vertex[index].pos.x << " " << rec_vertex[index].pos.y << " " << rec_vertex[index].pos.z << std::endl;
        //look if reconstructed vertex matches with reconstrutible by distance criterion
        std::cout <<"compare vertex positions: " << rec_vertex[index].pos.z << " " << vtx.z << " " << 5. * rec_vertex[index].cov[5] << std::endl;
        if(abs(rec_vertex[index].pos.z - vtx.z) <  5. * rec_vertex[index].cov[5]) {
        //if(abs(rec_vertex[index].pos.z - vtx.z) <  1.) {
          number_reconstructed_vertices++;

          continue;
        }
        else number_fake_vertices++;
        
      }
    }
    i_event++;
  }
  std::cout << "found " << number_reconstructed_vertices << " / " << number_reconstructible_vertices << " vertices!" << std::endl; 
  std::cout << "fakes: " << number_fake_vertices << std::endl;
*/

}
