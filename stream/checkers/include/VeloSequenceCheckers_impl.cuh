#include "ConsolidateVelo.cuh"

/**
 * @brief Specialization for any Velo reconstruction algorithm invoking
 *        consolidate_tracks_t as last step.
 */
template<>
void SequenceVisitor::check<consolidate_velo_tracks_t>(
  const uint& start_event_offset,
  const uint& number_of_events_requested,
  const HostBuffers& host_buffers,
  const Constants& constants,
  const CheckerInvoker& checker_invoker) const
{
  info_cout << "Checking GPU Velo tracks" << std::endl;

  const auto tracks = prepareVeloTracks(
    host_buffers.host_atomics_velo,
    host_buffers.host_velo_track_hit_number,
    host_buffers.host_velo_track_hits,
    number_of_events_requested);

  checker_invoker.check<TrackCheckerVelo>(
    start_event_offset,
    tracks);
}
