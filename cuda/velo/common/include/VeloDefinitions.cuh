#pragma once

namespace VeloTracking {

// Detector constants
static constexpr uint n_modules = 52;
static constexpr uint n_sensors = n_modules * 4;
static constexpr float z_endVelo = 770;

// How many concurrent h1s to process max
// It should be a divisor of NUMTHREADS_X
static constexpr uint max_concurrent_h1 = 16;

// Number of concurrent h1s in the first iteration
// The first iteration has no flagged hits and more triplets per hit
static constexpr uint max_concurrent_h1_first_iteration = 8;

// These parameters impact the found tracks
// Maximum / minimum acceptable phi
// This impacts enourmously the speed of track seeding
// static constexpr float phi_extrapolation = 0.0436332f;
static constexpr float phi_extrapolation = 0.062f;
// static constexpr float phi_extrapolation = 0.0959931f;

// Forward tolerance in phi
constexpr float forward_phi_tolerance = 0.052f;

// Tolerance angle for forming triplets
static constexpr float max_slope = 0.4f;
static constexpr float tolerance = 0.6f;

// Maximum scatter of each three hits
// This impacts velo tracks and a to a lesser extent
// long and long strange tracks
static constexpr float max_scatter_seeding = 0.004f;

// Making a bigger forwarding scatter window causes
// less clones and more ghosts
static constexpr float max_scatter_forwarding = 0.004f;
// Maximum number of skipped modules allowed for a track
// before storing it
static constexpr uint max_skipped_modules = 1;

// Total number of atomics required
// This is just a constant
static constexpr uint num_atomics = 5;

// Constants for requested storage on device
static constexpr uint max_tracks = 1200;
static constexpr uint max_track_size = 26;

// Constant for maximum number of hits in a module
static constexpr uint max_numhits_in_module = 500;

// Maximum number of tracks to follow at a time
static constexpr uint ttf_modulo = 2000;
static constexpr uint max_weak_tracks = 500;

// High number of hits per event
static constexpr uint max_number_of_hits_per_event = 9500;

// Constants for filters
static constexpr uint states_per_track = 3; 
static constexpr float param_w = 3966.94f;
static constexpr float param_w_inverted = 0.000252083f;

// Max chi2
static constexpr float max_chi2 = 20.0;

}
