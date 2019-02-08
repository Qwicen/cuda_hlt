#include "ParKalmanFilter.cuh"
#include "KalmanChecker.h"
#include "PrepareKalmanTracks.h"

template<>
void SequenceVisitor::check<kalman_filter_t>(
  const uint& start_event_offset,
  const uint& number_of_events_requested,
  const HostBuffers& host_buffers,
  const Constants& constants,
  const CheckerInvoker& checker_invoker) const
{
// Note: Nothing happens if not compiled with ROOT
#ifdef WITH_ROOT
  info_cout << "Producing Kalman plots" << std::endl << std::endl;

  const auto tracks = prepareKalmanTracks(
    host_buffers.host_atomics_velo,
    host_buffers.host_velo_track_hit_number,
    host_buffers.host_velo_track_hits,
    host_buffers.host_atomics_ut,
    host_buffers.host_ut_track_hit_number,
    host_buffers.host_ut_track_hits,
    host_buffers.host_ut_track_velo_indices,
    host_buffers.host_ut_qop,
    host_buffers.host_atomics_scifi,
    host_buffers.host_scifi_track_hit_number,
    host_buffers.host_scifi_track_hits,
    host_buffers.host_scifi_track_ut_indices,
    host_buffers.host_scifi_qop,
    host_buffers.host_scifi_states,
    constants.host_scifi_geometry,
    constants.host_inv_clus_res,
    host_buffers.host_kf_tracks,
    host_buffers.host_number_of_selected_events[0]);

  checkKalmanTracks(start_event_offset, tracks, checker_invoker.selected_mc_events);
#endif
}
