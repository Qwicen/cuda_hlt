#include "SequenceVisitor.cuh"
#include "fitSeeds.cuh"

template<>
void SequenceVisitor::set_arguments_size<fitSeeds_t>(
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
void SequenceVisitor::visit<fitSeeds_t>(
  fitSeeds_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Consolidate tracks
  // TODO: The size specified (sizeof(Hits) / sizeof(uint)) is due to the
  //       lgenfe error from the nvcc compiler, present in Cuda 9.2. Once it
  //       is gone, we can switch all pointers to char*.
 // arguments.set_size<arg::dev_velo_track_hits>(host_buffers.host_accumulated_number_of_hits_in_velo_tracks[0] * sizeof(Velo::Hit) / sizeof(uint));
 // arguments.set_size<arg::dev_velo_states>(host_buffers.host_number_of_reconstructed_velo_tracks[0] * sizeof(Velo::State) / sizeof(uint));


  state.set_opts(dim3(runtime_options.number_of_events), 1, cuda_stream);
  state.set_arguments(
    arguments.offset<dev_vertex>(),
    arguments.offset<dev_number_vertex>(),
    arguments.offset<dev_seeds>(),
    arguments.offset<dev_number_seeds>(),
    arguments.offset<dev_kalmanvelo_states>(),
    arguments.offset<dev_atomics_storage>(),
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

      // Wait to receive the result
  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);


  for(int i_event = 0; i_event < runtime_options.number_of_events; i_event++) {
    std::cout << "event " << i_event << " rec vtx: " << host_buffers.host_number_of_vertex[i_event] << std::endl;
    for(int i_vtx = 0; i_vtx < host_buffers.host_number_of_vertex[i_event]; i_vtx++) {
      int index = PatPV::max_number_vertices * i_event + i_vtx;
      std::cout <<i_event<< " vtx " <<  host_buffers.host_reconstructed_pvs[index].x << " " <<host_buffers.host_reconstructed_pvs[index].y << " " << host_buffers.host_reconstructed_pvs[index].z << " " << host_buffers.host_reconstructed_pvs[index].cov22 << std::endl;
    }
  }

    
}


