#include "pv_beamline_histo.cuh"

__device__ float gauss_integral(float x)
{
  const float a = std::sqrt(float(2 * order_polynomial + 3));
  const float xi = x / a;
  const float eta = 1.f - xi * xi;
  constexpr float p[] = {0.5f, 0.25f, 0.1875f, 0.15625f};
  // be careful: if you choose here one order more, you also need to choose 'a' differently (a(N)=sqrt(2N+3))
  return 0.5f + xi * (p[0] + eta * (p[1] + eta * p[2]));
}

__global__ void
pv_beamline_histo(int* dev_atomics_storage, uint* dev_velo_track_hit_number, PVTrack* dev_pvtracks, float* dev_zhisto)
{

  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;

  const Velo::Consolidated::Tracks velo_tracks {
    (uint*) dev_atomics_storage, dev_velo_track_hit_number, event_number, number_of_events};

  const uint number_of_tracks_event = velo_tracks.number_of_tracks(event_number);
  const uint event_tracks_offset = velo_tracks.tracks_offset(event_number);

  float* histo_base_pointer = dev_zhisto + Nbins * event_number;

  // find better wy to intialize histogram bins to zero
  if (threadIdx.x == 0) {
    for (int i = 0; i < Nbins; i++) {
      *(histo_base_pointer + i) = 0.f;
    }
  }
  __syncthreads();

  for (int i = 0; i < number_of_tracks_event / blockDim.x + 1; i++) {
    int index = blockDim.x * i + threadIdx.x;
    if (index < number_of_tracks_event) {

      PVTrack trk = dev_pvtracks[event_tracks_offset + index];
      // apply the z cut here
      if (zmin < trk.z && trk.z < zmax) {

        // bin in which z0 is, in floating point
        const float zbin = (trk.z - zmin) / dz;

        // to compute the size of the window, we use the track
        // errors. eventually we can just parametrize this as function of
        // track slope.
        const float zweight = trk.tx.x * trk.tx.x * trk.W_00 + trk.tx.y * trk.tx.y * trk.W_11;
        const float zerr = 1.f / std::sqrt(zweight);
        // get rid of useless tracks. must be a bit carefull with this.
        if (zerr < maxTrackZ0Err) { // m_nsigma < 10*m_dz ) {
          // find better place to define this
          const float a = std::sqrt(float(2 * order_polynomial + 3));
          const float halfwindow = a * zerr / dz;
          // this looks a bit funny, but we need the first and last bin of the histogram to remain empty.
          const int minbin = std::max(int(zbin - halfwindow), 1);
          const int maxbin = std::min(int(zbin + halfwindow), Nbins - 2);
          // we can get rid of this if statement if we make a selection of seeds earlier
          if (maxbin >= minbin) {
            float integral = 0;
            for (auto i = minbin; i < maxbin; ++i) {
              const float relz = (zmin + (i + 1) * dz - trk.z) / zerr;
              const float thisintegral = gauss_integral(relz);
              atomicAdd(histo_base_pointer + i, thisintegral - integral);
              integral = thisintegral;
            }
            // deal with the last bin
            atomicAdd(histo_base_pointer + maxbin, 1.f - integral);
          }
        }
      }
    }
  }
}
