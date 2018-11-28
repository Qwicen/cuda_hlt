#include "SequenceVisitor.cuh"
#include "getSeeds.cuh"

template<>
void SequenceVisitor::set_arguments_size<pv_get_seeds_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  // Set arguments size
  arguments.set_size<dev_seeds>(host_buffers.host_number_of_reconstructed_velo_tracks[0] );
  arguments.set_size<dev_number_seeds>(runtime_options.number_of_events );
}



template<>
void SequenceVisitor::visit<pv_get_seeds_t>(
  pv_get_seeds_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{

  state.set_opts(dim3(runtime_options.number_of_events), 1, cuda_stream);
  state.set_arguments(
    arguments.offset<dev_kalmanvelo_states>(),
    arguments.offset<dev_atomics_velo>(),
    arguments.offset<dev_velo_track_hit_number>(),
    arguments.offset<dev_seeds>(),
    arguments.offset<dev_number_seeds>()
  );


  state.invoke();


      cudaCheck(cudaMemcpyAsync(
    host_buffers.host_number_of_seeds,
    arguments.offset<dev_number_seeds>(),
    arguments.size<dev_number_seeds>(),
    cudaMemcpyDeviceToHost,
    cuda_stream
  ));





    
}
