#include "SequenceVisitor.cuh"
#include "VeloKalmanFilter.cuh"

template<>
void SequenceVisitor::visit<velo_fit_t>(
  velo_fit_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
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
  arguments.set_size<arg::dev_kalmanvelo_states>(host_buffers.host_number_of_reconstructed_velo_tracks[0] * sizeof(Velo::State) / sizeof(uint));
  scheduler.setup_next(arguments, sequence_step);

  state.set_opts(dim3(runtime_options.number_of_events), dim3(256), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_atomics_storage>(),
    arguments.offset<arg::dev_tracks>(),
    arguments.offset<arg::dev_velo_track_hit_number>(),
    arguments.offset<arg::dev_velo_track_hits>(),
    arguments.offset<arg::dev_velo_states>(),
    arguments.offset<arg::dev_kalmanvelo_states>()
  );

  state.invoke();



    
}



    // Check the output



  // TODO: Perhaps this shouldn't go here

  // Transmission device to host
  // Velo tracks
  

    

