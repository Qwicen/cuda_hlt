#include "SequenceVisitor.cuh"
#include "blpv_histo.cuh"

template<>
void SequenceVisitor::set_arguments_size<blpv_histo_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  // Set arguments size
  arguments.set_size<dev_zhisto>(runtime_options.number_of_events * (m_zmax-m_zmin)/m_dz);
}



template<>
void SequenceVisitor::visit<blpv_histo_t>(
  blpv_histo_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{

  state.set_opts(dim3(runtime_options.number_of_events), 100, cuda_stream);
  state.set_arguments(
    arguments.offset<dev_atomics_storage>(),
    arguments.offset<dev_velo_track_hit_number>(),
    arguments.offset<dev_pvtracks>(),
    arguments.offset<dev_zhisto>()
  );


  state.invoke();


    
}
