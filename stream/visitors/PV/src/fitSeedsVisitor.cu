#include "SequenceVisitor.cuh"
#include "fitSeeds.cuh"

template<>
void SequenceVisitor::set_arguments_size<pv_fit_seeds_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  // Set arguments size
  arguments.set_size<dev_vertex>(PatPV::max_number_vertices * runtime_options.number_of_events );
  arguments.set_size<dev_number_vertex>(runtime_options.number_of_events );
}



template<>
void SequenceVisitor::visit<pv_fit_seeds_t>(
  pv_fit_seeds_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{


  state.set_opts(dim3(runtime_options.number_of_events), 1, cuda_stream);
  state.set_arguments(
    arguments.offset<dev_vertex>(),
    arguments.offset<dev_number_vertex>(),
    arguments.offset<dev_seeds>(),
    arguments.offset<dev_number_seeds>(),
    arguments.offset<dev_kalmanvelo_states>(),
    arguments.offset<dev_atomics_velo>(),
    arguments.offset<dev_velo_track_hit_number>()
  );


  state.invoke();

    // Retrieve result
  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_reconstructed_pvs,
    arguments.offset<dev_vertex>(),
    arguments.size<dev_vertex>(),
    cudaMemcpyDeviceToHost,
    cuda_stream
  ));

    cudaCheck(cudaMemcpyAsync(
    host_buffers.host_number_of_vertex,
    arguments.offset<dev_number_vertex>(),
    arguments.size<dev_number_vertex>(),
    cudaMemcpyDeviceToHost,
    cuda_stream
  ));






    
}


