#define Test_Selector_cxx


#include <TH2.h>
#include <TStyle.h>
#include <iostream>


#include <assert.h>

#include <TTree.h>
#include <TFile.h>

#include "../include/Test_Selector.hpp"
#include "../include/MCParticle.h"
#include "../include/kalman.h"
#include "../include/VPHit.h"
#include "../include/FTHit.h"



using namespace std;
void Test_Selector::Begin(TTree * /*tree*/)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

}

void Test_Selector::SlaveBegin(TTree * /*tree*/)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

}



void Test_Selector::GetVeloTracks(int evtId) { 
  
  

  Long64_t maxEntries = fChain->GetTree()->GetEntries();
  vector<track_t> velo_tracks;
  vector<state_t> states;
  vector<State> seed_states;
  vector<Track*> seed_tracks;
  //cout << "start event loop" << endl;
  for(Long64_t entry = 0; entry< maxEntries ; ++entry){
    fChain->GetTree()->GetEntry(entry);
    if( p>0 && hasVelo){
      //for p>0 all MCParticles with at least 1 hit in one of the sub-detectors
      track_t track;
      //cout << "getting hit" << endl;
      //cout << "size " << (*Velo_x).size() << " " << Velo_x->size() << " " << (*Velo_x).at(0) <<  endl;
      vector<const hit_t * > hit_vector;
      //get all Velo hits for one MC particle
      for(int index = 0; index < Velo_x->size(); index++) {
        hit_t * hit = new hit_t;
        hit->x = Velo_x->at(index);
        hit->y = Velo_y->at(index);
        hit->z = Velo_z->at(index);
        //cout << hit->x << " " << hit->y << " " << hit->z<< endl;
        //cout << Velo_z->at(index) << endl;
        hit_vector.push_back(hit);
      }
      //apply Kalman fit
      //cout << "got hit" << endl;
      track.hits = hit_vector;
      state_t state;

      //go from last hit to hit closest to velo
      fitKalman(track, state, -1);
      finalizeKalman(state);
      //cout << state.x << " " << state.y << " " << state.z <<endl;  
      states.push_back(state);
      State seed_state;
      seed_state.tx = state.tx;
      seed_state.ty = state.ty;
      seed_state.x  = state.x;
      seed_state.y  = state.y;
      seed_state.z  = state.z;
      seed_state.errX2 = state.covXX;
      seed_state.errY2 = state.covYY;
      //cout<<"cov: " << seed_state.errX2 << " " << seed_state.errY2 << endl;
      seed_states.push_back(seed_state);
      Track * seed_track = new Track();
      std::vector<state_t> just_state;
      just_state.push_back(state);
      seed_track->states = just_state;
      seed_tracks.push_back(seed_track);

      //std::cout<<"before transport: " << state.x << " " << state.y << " " << state.z << " " << state.covXX << " " << state.covYY << std::endl;
      //state.linearTransportTo(0.);
      //std::cout<<"after transport: " << state.x << " " << state.y << " " << state.z << " " << state.covXX << " " << state.covYY << std::endl;

    }

  }

  PVSeedTool seedtool;
  XYZPoint beamspot{0.,0.,0.};
  auto seeds = seedtool.getSeeds(seed_tracks, beamspot);
  for(auto seed : seeds) cout << seed.x << " " << seed.y << " " << seed.z<< endl;

  AdaptivePV3DFitter fitter;
  std::vector<Track*> tracks_to_remove;
  Vertex vtx;
  for(auto seed : seeds) {cout << seed_tracks.size() << endl;
  cout <<"success: " << fitter.fitVertex(seed, seed_tracks, vtx, tracks_to_remove) << endl;}
  

  //return states;
}

/*
std::vector<UTHit> Test_Selector::GetUTHits( mcparticleID){
   ---> loop over entries[mcparticle]
  ---> extract only the UTHits on this MCParticle
}


In the executable
VeloTrack = Test_Selector::GetVeloTracks();
for( auto & velotr: VeloTracks){
   UTHits = GetUTHits( velotr.AssociatedParticle);
  --> do your study of UTHit - predicted Velo track position )

*/



Bool_t Test_Selector::Process( Long64_t entry){
  return 1;
}

Bool_t Test_Selector::Process(){
  Long64_t maxEntries = fChain->GetTree()->GetEntries();

  vector<MCParticle> MCParticles;
  MCParticles.reserve( fChain->GetTree()->GetEntries() );
  
  
  vector<unsigned int > velo_lhcbids;
  velo_lhcbids.reserve( 9000);
  //dummy vector container of Velo hits, can be a array<vector<VPHit>, Modules> or whatever you want
  vector<VPHit> VeloHits;
  VeloHits.reserve(9000);
  
  vector<unsigned int > ut_lhcbids;
  ut_lhcbids.reserve(5000);
  //scifi 
  vector<unsigned int > scifi_lhcbids;
  scifi_lhcbids.reserve( 5000);
  vector<FTHit> FTHits;
  FTHits.reserve(5000);

  int nMCParticles = 0;
  for(Long64_t entry = 0; entry< maxEntries ; ++entry){
    fChain->GetTree()->GetEntry(entry);
    if( p>0 ){
      //for p>0 all MCParticles with at least 1 hit in one of the sub-detectors
      nMCParticles++;
      MCParticle currMCParticle;
      currMCParticle.setNbHitsAssociated( nVeloHits, nUTHits ,  nFTHits);
      currMCParticle.setKinematic( p, pt, eta);
      currMCParticle.setVertexPosition( ovtx_x, ovtx_y, ovtx_z);
      currMCParticle.setflagsDetectors( hasVelo, hasUT, hasSciFi);
      currMCParticle.setProperty( pid, DecayOriginMother_pid, fromBeautyDecay, fromCharmDecay, fromStrangeDecay) ;
      MCParticles.push_back( currMCParticle);
    }
    //get for that MCParticles all the FTHits associated
    for( int i =0; i< nFTHits; ++i){  
      auto lhcbid = (*FT_lhcbID)[i];
      if( std::find( scifi_lhcbids.begin(), scifi_lhcbids.end(), lhcbid) != scifi_lhcbids.end() ){
	//already filled, this hit is already linekd to another MCParticle.... we may want to keep this info for the FTHit key
	//or later in truth matching ( 3 MCParticles to 1 hit, we use only 1 MCParticle for the association ,
	//we could be a bit smarter...
	continue;
      }else{
	scifi_lhcbids.push_back( lhcbid);
      }
      //define your own constructor for the FTHits
      auto fthit = FTHit( (*FT_lhcbID)[i],
			  (*FT_x)[i],
			  (*FT_z)[i],
			  (*FT_dxdy)[i],
			  (*FT_hitPlaneCode)[i],
			  (*FT_hitzone)[i]);
      if( p>0){
	fthit.setMCParticleKey( key);
      }else{
	fthit.setMCParticleKey( -99999);
      }
      //whatever, p>0 or p<0
      FTHits.push_back( fthit);
    }
    
    
    //Deal with  velo hits
    for( int i =0; i< nVeloHits ; ++i){
      auto lhcbid = (*Velo_lhcbID)[i];
      if( std::find(velo_lhcbids.begin(), velo_lhcbids.end(), lhcbid) != velo_lhcbids.end() ){
	//do not fill 2 times the same velo hit in the velo hit container for this event
	continue;
	}else{
	velo_lhcbids.push_back( lhcbid);
      }
      auto VeloHit = VPHit( (*Velo_lhcbID)[i],
			    (*Velo_x)[i],
			    (*Velo_y)[i],
			    (*Velo_z)[i],
			    (*Velo_Module)[i]);
      if( p>0){
	VeloHit.setMCParticleKey(key);
      }else{
	VeloHit.setMCParticleKey(-99999);
      }
      VeloHits.push_back( VeloHit);
    }
  }

  std::cout<<"Nb MC particles in the event " << nMCParticles<<std::endl;
  std::cout<<"Nb MC particles in the event  "<< MCParticles.size()<<std::endl;


  
  std::cout<<"Nb unique FT hits in event " << FTHits.size()<<std::endl;
  std::cout<<"Nb unique Velo hits in event " << VeloHits.size()<<std::endl;

  
  int nElectrons = std::count_if( MCParticles.begin(), MCParticles.end() , [](MCParticle & mcp){ return mcp.isElectron( );});
  int nPions     = std::count_if( MCParticles.begin(), MCParticles.end() , [](MCParticle & mcp){ return mcp.isPion( );});
  int nKaons     = std::count_if( MCParticles.begin(), MCParticles.end() , [](MCParticle & mcp){ return mcp.isKaon( );});
  int nMuon      = std::count_if( MCParticles.begin(), MCParticles.end() , [](MCParticle & mcp){ return mcp.isMuon( );});
  int nProtons   = std::count_if( MCParticles.begin(), MCParticles.end() , [](MCParticle & mcp){ return mcp.isProton( );});

  //also sigmas should be included....anyway... fair enough
  int nOthers = std::count_if( MCParticles.begin(), MCParticles.end() , [](MCParticle & mcp){ return mcp.isDeutTritHe3Alpha4( );});

  std::cout<<" MC particle event composition : \n \t Electrons "<< nElectrons << "\n \t Pions "<< nPions << "\n \t Kaons " << nKaons << "\n \t Muons "<< nMuon << "\n \t Protons " << nProtons <<"\n \t Deut/Trit/He3/Alpha "<< nOthers<< "\n ----- Sum = "<< nElectrons + nPions + nKaons + nMuon + nProtons + nOthers << std::endl;



  return kTRUE;
}

void Test_Selector::SlaveTerminate()
{
   // The SlaveTerminate() function is called after all entries or objects
   // have been processed. When running with PROOF SlaveTerminate() is called
   // on each slave server.

}

void Test_Selector::Terminate()
{
   // The Terminate() function is the last function to be called during
   // a query. It always runs on the client, it can be used to present
   // the results graphically or save the results to file.

}
