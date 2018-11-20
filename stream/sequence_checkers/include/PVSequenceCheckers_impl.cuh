#include "PrimaryVertexChecker.h"

/**
 * @brief Specialization for patPV PV finding algorithm
 */
template<>
void SequenceVisitor::check<fitSeeds_t>(
  const uint& start_event_offset,
  const uint& number_of_events_requested,
  const HostBuffers& host_buffers,
  const CheckerInvoker& checker_invoker) const
{
  info_cout << "Checking GPU PVs " << checker_invoker.mc_pv_folder << std::endl;
  checkPVs( checker_invoker.mc_pv_folder,  number_of_events_requested, host_buffers.host_reconstructed_pvs, host_buffers.host_number_of_vertex);
  
}
 

/**
 * @brief Specialization for beamline PV finding algorithm
 */
template<>
void SequenceVisitor::check<cpu_beamlinePV_t>(
  const uint& start_event_offset,
  const uint& number_of_events_requested,
  const HostBuffers& host_buffers,
  const CheckerInvoker& checker_invoker) const
{
  info_cout << "Checking CPU beamline PVs " << checker_invoker.mc_pv_folder << std::endl;
  checkPVs( checker_invoker.mc_pv_folder,  number_of_events_requested, host_buffers.host_reconstructed_pvs, host_buffers.host_number_of_vertex);
  
}
  

