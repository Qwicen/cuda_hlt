#pragma once

unsigned int m_minNumTracksPerVertex = 5;
float        m_zmin = - 300.; //unit: mm Min z position of vertex seed
float        m_zmax = 300; //unit: mm Max z position of vertex seed
float        m_dz = 0.25; //unit: mm Z histogram bin size
float        m_maxTrackZ0Err = 1.5; // unit: mm "Maximum z0-error for adding track to histo"
float        m_minDensity = 1.0; // unit: 1./mm "Minimal density at cluster peak  (inverse resolution)"
float        m_minDipDensity = 2.0; // unit: 1./mm,"Minimal depth of a dip to split cluster (inverse resolution)"
float        m_minTracksInSeed = 2.5; // "MinTrackIntegralInSeed"
float        m_maxVertexRho = 0.3; // unit:: mm "Maximum distance of vertex to beam line" 
unsigned int m_maxFitIter = 5; // "Maximum number of iterations for vertex fit"
float        m_maxDeltaChi2 = 9; //"Maximum chi2 contribution of track to vertex fit"

// Get the beamline. this only accounts for position, not
// rotation. that's something to improve! 
 
// set this to (0,0) for now
const float2 beamline{0.f, 0.f};
 
