// *********************************************************************************
// ************************ Introduction to Forward Tracking **********************
// *********************************************************************************
//
//  A detailed introduction in Forward tracking (with real pictures!) can be
//  found here:
//  (2002) http://cds.cern.ch/record/684710/files/lhcb-2002-008.pdf
//  (2007) http://cds.cern.ch/record/1033584/files/lhcb-2007-015.pdf
//  (2014) http://cds.cern.ch/record/1641927/files/LHCb-PUB-2014-001.pdf
//
// *** Short Introduction in geometry:
//
// The SciFi Tracker Detector, or simple Fibre Tracker (FT) consits out of 3 stations.
// Each station consists out of 4 planes/layers. Thus there are in total 12 layers,
// in which a particle can leave a hit. The reasonable maximum number of hits a track
// can have is thus also 12 (sometimes 2 hits per layer are picked up).
//
// Each layer consists out of several Fibre mats. A fibre has a diameter of below a mm.(FIXME)
// Several fibres are glued alongside each other to form a mat.
// A Scintilating Fibre produces light, if a particle traverses. This light is then
// detected on the outside of the Fibre mat.
//
// Looking from the collision point, one (X-)layer looks like the following:
//
//                    y       6m
//                    ^  ||||||||||||| Upper side
//                    |  ||||||||||||| 2.5m
//                    |  |||||||||||||
//                   -|--||||||o||||||----> -x
//                       |||||||||||||
//                       ||||||||||||| Lower side
//                       ||||||||||||| 2.5m
//
// All fibres are aranged parallel to the y-axis. There are three different
// kinds of layers, denoted by X,U,V. The U/V layers are rotated with respect to
// the X-layers by +/- 5 degrees, to also get a handle of the y position of the
// particle. As due to the magnetic field particles are only deflected in
// x-direction, this configuration offers the best resolution.
// The layer structure in the FT is XUVX-XUVX-XUVX.
//
// The detector is divided into an upeer and a lower side (>/< y=0). As particles
// are only deflected in x direction there are only very(!) few particles that go
// from the lower to the upper side, or vice versa. The reconstruction algorithm
// can therefore be split into two independent steps: First track reconstruction
// for tracks in the upper side, and afterwards for tracks in the lower side.
//
// Due to construction issues this is NOT true for U/V layers. In these layers the
// complete(!) fibre modules are rotated, producing a zic-zac pattern at y=0, also
// called  "the triangles". Therefore for U/V layers it must be explicetly also
// searched for these hit on the "other side", if the track is close to y=0.
// Sketch (rotation exagerated!):
//                                          _.*
//     y ^   _.*                         _.*
//       | .*._      Upper side       _.*._
//       |     *._                 _.*     *._
//       |--------*._           _.*           *._----------------> x
//       |           *._     _.*                 *._     _.*
//                      *._.*       Lower side      *._.*
//
//
//
//
//
//       Zone ordering defined on PrKernel/PrFTInfo.h
//
//     y ^
//       |    1  3  5  7     9 11 13 15    17 19 21 23
//       |    |  |  |  |     |  |  |  |     |  |  |  |
//       |    x  u  v  x     x  u  v  x     x  u  v  x   <-- type of layer
//       |    |  |  |  |     |  |  |  |     |  |  |  |
//       |------------------------------------------------> z
//       |    |  |  |  |     |  |  |  |     |  |  |  |
//       |    |  |  |  |     |  |  |  |     |  |  |  |
//       |    0  2  4  6     8 10 12 14    16 18 20 22
//
//
// *** Short introduction in the Forward Tracking algorithm
//
// The track reconstruction is seperated into several steps:
//
// 1) Using only X-hits
//    1.1) Preselection: collectAllXHits()
//    1.2) Hough Transformation: xAtRef_SamePlaneHits()
//    1.3) Cluster search: selectXCandidates()
//    1.4) Linear and than Cubic Fit of X-Projection
// 2) Introducing U/V hits or also called stereo hits
//    2.1) Preselection: collectStereoHits
//    2.2) Cluster search: selectStereoHits
//    2.3) Fit Y-Projection
// 3) Using all (U+V+X) hits
//    3.1) Fitting X-Projection
//    3.2) calculating track quality with a Neural Net
//    3.3) final clone+ghost killing
//
// *****************************************************************

#include "PrForward.cuh"

//-----------------------------------------------------------------------------
// Implementation file for class : PrForward
//
// Based on code written by :
// 2012-03-20 : Olivier Callot
// 2013-03-15 : Thomas Nikodem
// 2015-02-13 : Sevda Esen [additional search in the triangles by Marian Stahl]
// 2016-03-09 : Thomas Nikodem [complete restructuring]
// 2018-08    : Vava Gligorov [extract code from Rec, make compile within GPU framework
// 2018-09    : Dorothea vom Bruch [convert to CUDA, runs on GPU]
//-----------------------------------------------------------------------------

//=============================================================================

// Kernel to call Forward tracking on GPU
// Loop over veloUT input tracks using threadIdx.x
__global__ void scifi_pr_forward(
  uint32_t* dev_scifi_hits,
  const uint32_t* dev_scifi_hit_count,
  const int* dev_atomics_velo,
  const uint* dev_velo_track_hit_number,
  const char* dev_velo_states,
  const int* dev_atomics_ut,
  const char* dev_ut_track_hits,
  const uint* dev_ut_track_hit_number,
  const float* dev_ut_qop,
  const uint* dev_ut_track_velo_indices,
  SciFi::TrackHits* dev_scifi_tracks,
  int* dev_atomics_scifi,
  const SciFi::Tracking::TMVA* dev_tmva1,
  const SciFi::Tracking::TMVA* dev_tmva2,
  const SciFi::Tracking::Arrays* dev_constArrays,
  const char* dev_scifi_geometry,
  const float* dev_inv_clus_res)
{
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;

  // Velo consolidated types
  const Velo::Consolidated::Tracks velo_tracks {
    (uint*) dev_atomics_velo, (uint*) dev_velo_track_hit_number, event_number, number_of_events};
  const Velo::Consolidated::States velo_states {(char*) dev_velo_states, velo_tracks.total_number_of_tracks};
  const uint velo_tracks_offset_event = velo_tracks.tracks_offset(event_number);

  // UT consolidated tracks
  UT::Consolidated::Tracks ut_tracks {(uint*) dev_atomics_ut,
                                      (uint*) dev_ut_track_hit_number,
                                      (float*) dev_ut_qop,
                                      (uint*) dev_ut_track_velo_indices,
                                      event_number,
                                      number_of_events};
  const int n_veloUT_tracks_event = ut_tracks.number_of_tracks(event_number);

  // SciFi un-consolidated track types
  SciFi::TrackHits* scifi_tracks_event = dev_scifi_tracks + event_number * SciFi::Constants::max_tracks;
  int* atomics_scifi_event = dev_atomics_scifi + event_number;

  // SciFi hits
  const uint total_number_of_hits = dev_scifi_hit_count[number_of_events * SciFi::Constants::n_mat_groups_and_mats];
  const SciFi::HitCount scifi_hit_count {(uint32_t*) dev_scifi_hit_count, event_number};
  const SciFi::SciFiGeometry scifi_geometry {dev_scifi_geometry};
  SciFi::Hits scifi_hits(dev_scifi_hits, total_number_of_hits, &scifi_geometry, dev_inv_clus_res);

  // initialize atomic SciFi tracks counter
  if (threadIdx.x == 0) {
    *atomics_scifi_event = 0;
  }
  __syncthreads();

  // Loop over the veloUT input tracks
  for (int i = 0; i < (n_veloUT_tracks_event + blockDim.x - 1) / blockDim.x; ++i) {
    const int i_veloUT_track = i * blockDim.x + threadIdx.x;
    if (i_veloUT_track < n_veloUT_tracks_event) {
      const float qop_ut = ut_tracks.qop[i_veloUT_track];

      const int i_velo_track = ut_tracks.velo_track[i_veloUT_track];
      const uint velo_states_index = velo_tracks_offset_event + i_velo_track;
      const MiniState velo_state {velo_states, velo_states_index};

      find_forward_tracks(
        scifi_hits,
        scifi_hit_count,
        qop_ut,
        i_veloUT_track,
        scifi_tracks_event,
        (uint*) atomics_scifi_event,
        dev_tmva1,
        dev_tmva2,
        dev_constArrays,
        velo_state);
    }
  }
}
