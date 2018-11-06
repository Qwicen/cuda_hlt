#include "SequenceVisitor.cuh"
#include "getSeeds.cuh"

template<>
void SequenceVisitor::set_arguments_size<getSeeds_t>(
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
void SequenceVisitor::visit<getSeeds_t>(
  getSeeds_t& state,
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
    arguments.offset<dev_kalmanvelo_states>(),
    arguments.offset<dev_atomics_storage>(),
    arguments.offset<dev_velo_track_hit_number>(),
    arguments.offset<dev_seeds>(),
    arguments.offset<dev_number_seeds>()
  );


  state.invoke();


    
}
