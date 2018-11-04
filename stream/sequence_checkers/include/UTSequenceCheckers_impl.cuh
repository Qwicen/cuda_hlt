#include "VeloUT.cuh"

/**
 * @brief Specialization for any Velo reconstruction algorithm invoking
 *        veloUT_t as last step.
 */
template<>
void SequenceVisitor::check<veloUT_t>(
  const uint& start_event_offset,
  const uint& number_of_events_requested,
  const HostBuffers& host_buffers,
  const PrCheckerInvoker& pr_checker_invoker) const
{
  info_cout << "Checking " << veloUT_t::name << " tracks" << std::endl;

  const auto tracks = prepareTracks<TrackCheckerVeloUT>(
    host_buffers.host_veloUT_tracks,
    host_buffers.host_atomics_veloUT,
    number_of_events_requested);

  pr_checker_invoker.check<TrackCheckerVeloUT>(
    start_event_offset,
    tracks);
}
