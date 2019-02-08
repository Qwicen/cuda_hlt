#include "PrForward.cuh"
#include "RunForwardCPU.h"

/**
 * @brief Specialization when invoking scifi_pr_forward_t as last step.
 */
template<>
void SequenceVisitor::check<consolidate_scifi_tracks_t>(
  const uint& start_event_offset,
  const uint& number_of_events_requested,
  const HostBuffers& host_buffers,
  const Constants& constants,
  const CheckerInvoker& checker_invoker) const
{
  info_cout << "Checking SciFi tracks" << std::endl;

  const auto tracks = prepareSciFiTracks(
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
    host_buffers.host_number_of_selected_events[0]);

  checker_invoker.check<TrackCheckerForward>(start_event_offset, tracks);
}
