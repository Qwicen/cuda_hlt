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

      if(mc_vertex.numberTracks >= 4) vertices.push_back(mc_vertex);
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


    events_vertices.push_back(vertices);
  } //end loop over files/events
  

  std::cout << "found " << number_reconstructed_vertices << " / " << number_reconstructible_vertices << " vertices! -> efficiency: " << (double)number_reconstructed_vertices / (double)number_reconstructible_vertices << std::endl; 
  std::cout << "fakes: " << number_fake_vertices << std::endl;
    
    
    
 

}
