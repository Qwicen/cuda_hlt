#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

int main ( ) {

  std::vector<uint> reco_ids;
  std::vector<uint> mc_ids;

  std::ifstream reco_in ("first_event_reco_tracks_from_input.txt" );
  std::ifstream mc_in ("first_event_MC_tracks.txt" );

  std::string line;
  while ( getline( reco_in, line ) ) {
    cout << "reco line = " << line << endl;
    uint id = std::stoi( line, nullptr, 0 );
    reco_ids.push_back( id );
    //printf("id = %x \n", id);
  }
  while ( getline( mc_in, line ) ) {
    cout << "mc line = " << line << endl;
    uint id = std::stoi( line, nullptr, 0 );
    mc_ids.push_back( id );
    //printf("id = %x \n", id);
  }

  for ( auto id : reco_ids ) {
    printf("checking id %u \n", id );
    std::vector<uint>::iterator it = std::find( mc_ids.begin(), mc_ids.end(), id );
    if ( it != mc_ids.end() )
      printf("Found match: %u = %u \n", *it, id);
  }
  
  return 0;
}
