#pragma once

#include "TrackChecker.h"

#ifdef WITH_ROOT
#include "TH1D.h"
#endif

template< typename t_checker >
struct Histos {
  
#ifdef WITH_ROOT
  
  std::map< std::string, TH1D > h_reconstructible_eta;
  
  Histos( std::vector< typename t_checker::TrackHistos > trackHistos ) {
    for ( auto histo : trackHistos ) {
      const std::string category = histo.m_name;
      std::string name = category + "_Eta_reconstructible";
      h_reconstructible_eta[name] = TH1D(name.c_str(), name.c_str(), 0, 7, 50);
    }
  }


#endif
};
