#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdint.h>
#include <string>

#include "Tracks.h"

std::vector< Tracks > read_input_tracks ( std::ifstream& input_tracks ) {

  std::vector< Tracks > all_tracks; // all tracks from all events

  unsigned long i_event = 0;
  unsigned long n_tracks = 0;

  std::string line;
  
  
  while ( !input_tracks.eof() ) {
    /* Get event index and # of tracks for that event */

    while ( getline( input_tracks, line ) ) {
      i_event = std::stoul( line, nullptr, 0);
      getline( input_tracks, line );
      n_tracks = std::stoul( line, nullptr, 0);
      
      printf("reading event %lu, with %lu tracks \n", i_event, n_tracks);
      Tracks tracks; // all tracks within one event
      for ( int i_track = 0; i_track < n_tracks; ++i_track ) {
	Track t;
	unsigned long n_hits;
	getline( input_tracks, line);
	n_hits = std::stoul( line, nullptr, 0);
	//printf("   track has %lu hits\n", n_hits);
	for ( int i_hit = 0; i_hit < n_hits; ++i_hit ) {
	  unsigned long id;
	  getline( input_tracks, line );
	  id = std::stoul( line, nullptr, 0 );
	  //printf("      hit has id %lu \n", id);
	  LHCbID lhcb_id( id );
	  t.addId( lhcb_id );
	} // hits
	tracks.push_back( t );
      } // tracks
      all_tracks.push_back( tracks );
    } // events
  }
  
  return all_tracks;
}
