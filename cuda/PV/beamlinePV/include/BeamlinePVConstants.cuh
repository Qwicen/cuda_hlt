#pragma once

static constexpr unsigned int m_minNumTracksPerVertex = 5;
static constexpr float m_zmin = -300.f;        // unit: mm Min z position of vertex seed
static constexpr float m_zmax = 300.f;         // unit: mm Max z position of vertex seed
static constexpr float m_dz = 0.25f;           // unit: mm Z histogram bin size
static constexpr float m_maxTrackZ0Err = 1.5f; // unit: mm "Maximum z0-error for adding track to histo"
static constexpr float m_minDensity = 1.0f;    // unit: 1./mm "Minimal density at cluster peak  (inverse resolution)"
static constexpr float m_minDipDensity =
  2.0f; // unit: 1./mm,"Minimal depth of a dip to split cluster (inverse resolution)"
static constexpr float m_minTracksInSeed = 2.5f; // "MinTrackIntegralInSeed"
static constexpr float m_maxVertexRho = 0.3f;    // unit:: mm "Maximum distance of vertex to beam line"
static constexpr float m_maxVertexRho2 = 0.09f;
static constexpr unsigned int m_maxFitIter = 5; // "Maximum number of iterations for vertex fit"
static constexpr float m_maxDeltaChi2 = 9.f;    //"Maximum chi2 contribution of track to vertex fit"
static constexpr float m_minWeight = 0.3f;      //"minimum value of adaptive weight of track in multi fit"

// Get the beamline. this only accounts for position, not
// rotation. that's something to improve!

// set this to (0,0) for now
const float2 beamline {0.f, 0.f};
