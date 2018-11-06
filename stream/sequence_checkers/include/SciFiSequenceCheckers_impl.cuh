#include "PrForward.cuh"
#include "RunForwardCPU.h"

/**
 * @brief Specialization for any Velo reconstruction algorithm invoking
 *        scifi_pr_forward_t as last step.
 */
template<>
void SequenceVisitor::check<scifi_pr_forward_t>(
  const uint& start_event_offset,
  const uint& number_of_events_requested,
  const HostBuffers& host_buffers,
  const CheckerInvoker& checker_invoker) const
{
  info_cout << "Checking " << scifi_pr_forward_t::name << " tracks" << std::endl;

  const auto tracks = prepareTracks<TrackCheckerForward, SciFi::Track>(
    host_buffers.host_scifi_tracks,
    (const int*) host_buffers.host_n_scifi_tracks,
    number_of_events_requested
  );
  
  checker_invoker.check<TrackCheckerForward>(
    start_event_offset,
    tracks);
}

/**
 * @brief Specialization for any Velo reconstruction algorithm invoking
 *        cpu_scifi_pr_forward_t as last step.
 */
template<>
void SequenceVisitor::check<cpu_scifi_pr_forward_t>(
  const uint& start_event_offset,
  const uint& number_of_events_requested,
  const HostBuffers& host_buffers,
  const CheckerInvoker& checker_invoker) const
{
  info_cout << "Checking " << cpu_scifi_pr_forward_t::name << " tracks" << std::endl;
  
  checker_invoker.check<TrackCheckerForward>(
    start_event_offset,
    host_buffers.forward_tracks_events);
}
