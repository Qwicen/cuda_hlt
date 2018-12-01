#include "PrimaryVertexChecker.h"

/**
 * @brief Specialization for patPV PV finding algorithm
 */
template<>
void SequenceVisitor::check<pv_fit_seeds_t>(
  const uint& start_event_offset,
  const uint& number_of_events_requested,
  const HostBuffers& host_buffers,
  const Constants& constants,
  const CheckerInvoker& checker_invoker) const
{

  info_cout << "Checking GPU PVs " << checker_invoker.mc_pv_folder << std::endl;
  checkPVs(
    checker_invoker.mc_pv_folder,
    number_of_events_requested,
    host_buffers.host_reconstructed_pvs,
    host_buffers.host_number_of_vertex,
    "GPU");
}

/**
 * @brief Specialization for beamline PV finding algorithm
 */
template<>
void SequenceVisitor::check<cpu_beamlinePV_t>(
  const uint& start_event_offset,
  const uint& number_of_events_requested,
  const HostBuffers& host_buffers,
  const Constants& constants,
  const CheckerInvoker& checker_invoker) const
{
  /*
  for(int i_event = 0; i_event < number_of_events_requested; i_event++ ) {
  std::cout << "event " << i_event << std::endl;
    for(int i = 0; i < host_buffers.host_number_of_vertex[i_event]; i++ ) {
      PV::Vertex vertex = host_buffers.host_reconstructed_pvs[i*i_event * PV::max_number_vertices + i];
      std::cout << "----" << std::endl;
      std::cout << vertex.position.x << " " << vertex.cov00 << std::endl;
      std::cout << vertex.position.y << " " << vertex.cov11 << std::endl;
      std::cout << vertex.position.z << " " << vertex.cov22 << std::endl;
    }
  }
  */
  info_cout << "Checking CPU beamline PVs " << checker_invoker.mc_pv_folder << std::endl;
  checkPVs(
    checker_invoker.mc_pv_folder,
    number_of_events_requested,
    host_buffers.host_reconstructed_pvs,
    host_buffers.host_number_of_vertex,
    "CPU");
}

/**
 * @brief Specialization for beamline PV finding algorithm on GPU
 */
template<>
void SequenceVisitor::check<blpv_multi_fitter_t>(
  const uint& start_event_offset,
  const uint& number_of_events_requested,
  const HostBuffers& host_buffers,
  const Constants& constants,
  const CheckerInvoker& checker_invoker) const
{
  /*
  for(int i_event = 0; i_event < number_of_events_requested; i_event++ ) {
    std::cout << "event " << i_event << std::endl;
      for(int i = 0; i < host_buffers.host_number_of_multivertex[i_event]; i++ ) {
        PV::Vertex vertex = host_buffers.host_reconstructed_multi_pvs[i*i_event * PV::max_number_vertices + i];
        std::cout << "----" << std::endl;
        std::cout << vertex.position.x << " " << vertex.cov00 << std::endl;
        std::cout << vertex.position.y << " " << vertex.cov11 << std::endl;
        std::cout << vertex.position.z << " " << vertex.cov22 << std::endl;
      }
    }
    */
  info_cout << "Checking GPU beamline PVs " << checker_invoker.mc_pv_folder << std::endl;
  checkPVs(
    checker_invoker.mc_pv_folder,
    number_of_events_requested,
    host_buffers.host_reconstructed_multi_pvs,
    host_buffers.host_number_of_multivertex,
    "GPU");
}
