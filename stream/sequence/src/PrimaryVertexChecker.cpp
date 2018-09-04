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
    std::vector<MCVertex> vertices;
  
  for (uint i=0; i<requestedFiles; ++i) {
    // Read event #i in the list and add it to the inputs
    // if more files are requested than present in folder, read them again
    std::string readingFile = folderContents[i % folderContents.size()];
    std::string filename = foldername + "/" + readingFile;
    std::vector<char> inputContents;
    readFileIntoVector(foldername + "/" + readingFile, inputContents);

    uint8_t* input = (uint8_t*) inputContents.data();

    int number_mcpv = *((int*)  input); input += sizeof(int);
    //std::cout << "num MCPs = " << number_mcp << std::endl;
    for (uint32_t i=0; i<number_mcpv; ++i) {
      MCVertex vertex;
      int VertexNumberOfTracks = *((int*)  input); input += sizeof(int);
      if(VertexNumberOfTracks >= 4) number_reconstructible_vertices++;
      vertex.numberTracks = VertexNumberOfTracks;
      vertex.x = *((double*)  input); input += sizeof(double);
      vertex.y = *((double*)  input); input += sizeof(double);
      vertex.z = *((double*)  input); input += sizeof(double);

      if(vertex.numberTracks >= 4) vertices.push_back(vertex);

    }








    events_vertices.push_back(vertices);
  }
  /*
  std::cout << "here are the MC vertices!" << std::endl;
  for(auto vtx_vec : events_vertices) {
    for (auto vtx : vtx_vec) {
      std::cout << "x: " << vtx.x << std::endl;
        std::cout << "y: " << vtx.y << std::endl;
        std::cout << "z: " << vtx.z << std::endl;
        std::cout << "number of tracks: " << vtx.numberTracks << std::endl;
    }
  }*/

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
