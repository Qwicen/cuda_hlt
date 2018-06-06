#include "../../include/Tracks.h"

#include "PrVUTTrack.h"

int main() {

  // Create fake tracks
  const int nb_tracks = 10;
  const int nb_states = 20;

  Tracks tracks;
  for (int i=0; i<nb_tracks; ++i) {
    Track tr;
    for (int j=0; j<nb_states; ++j) {
      VeloState st;
      tr.emplace_back(st);
    }
    tracks.emplace_back(tr);
  }

  // Call the veloUT
  PrVeloUT velout;
  if ( velout::initialize() ) {
    velout(tracks);    
  }
 
}