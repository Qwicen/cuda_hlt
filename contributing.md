Cuda HLT: Adding a new CUDA algorithm
=====================================

This tutorial will guide you through adding a new CUDA algorithm to the `cuda_hlt` project.

SAXPY
-----

Writing an algorithm in CUDA in the `cuda_hlt` project is no different than writing it on any other GPU project. The differences are in how to invoke that program, and how to setup the options, arguments, and so on.

So let's assume that we have the following simple `SAXPY` algorithm, taken out from this website https://devblogs.nvidia.com/easy-introduction-cuda-c-and-c/

```clike=
__global__ void saxpy(float *x, float *y, int n, float a) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}
```

### Adding the CUDA algorithm

We want to add the algorithm to a specific folder inside the `cuda` folder:

```
├── cuda
│   ├── CMakeLists.txt
│   └── velo
│       ├── CMakeLists.txt
│       ├── calculate_phi_and_sort
│       │   ├── include
│       │   │   └── CalculatePhiAndSort.cuh
│       │   └── src
│       │       ├── CalculatePhiAndSort.cu
│       │       ├── CalculatePhi.cu
│       │       └── SortByPhi.cu
│       ├── common
│       │   ├── include
│       │   │   ├── ClusteringDefinitions.cuh
│       │   │   └── VeloDefinitions.cuh
│       │   └── src
│       │       ├── ClusteringDefinitions.cu
│       │       └── Definitions.cu
...
```

Let's create a new folder inside the `cuda` directory named `test`. We need to modify `cuda/CMakeLists.txt` to reflect this:

```cmake=
add_subdirectory(velo)
add_subdirectory(test)
```

Inside the `test` folder we will create the following structure:

```
├── test
│   ├── CMakeLists.txt
│   └── saxpy
│       ├── include
│       │   └── Saxpy.cuh
│       └── src
│           └── Saxpy.cu
```

The newly created `test/CMakeLists.txt` file should reflect the project we are creating. We can do that by populating it like so:

```cmake=
file(GLOB test_saxpy "saxpy/src/*cu")
include_directories(saxpy/include)
include_directories(../../stream/handlers/include)

cuda_add_library(Test STATIC
  ${test_saxpy}
)
```

Our CUDA algorithm `Saxpy.cuh` and `Saxpy.cu` will be as follows:

```clike=
#include "Handler.cuh"

__global__ void saxpy(float *x, float *y, int n, float a);

ALGORITHM(saxpy, saxpy_t)
```

```clike=
#include "Saxpy.cuh"

__global__ void saxpy(float *x, float *y, int n, float a) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}
```

The line with `ALGORITHM` encapsulates our algorithm `saxpy` into a class with name `saxpy_t`. We will use this class from now on to be able to refer to our algorithm.

Lastly, edit `stream/CMakeLists.txt` and modify `target_link_libraries`:

```cmake
target_link_libraries(Stream Velo Test)
```

Ready to move on.

### Integrating the algorithm in the sequence

`cuda_hlt` centers around the idea of running a __sequence of algorithms__ on input events. This sequence is predefined and will always be executed in the same order.

Some events from the input will be discarded throughout the execution, and only a fraction of them will be kept for further processing. That is conceptually the idea behind the _High Level Trigger 1_ stage of LHCb, and is what is intended to achieve with this project.

Therefore, we need to add our algorithm to the sequence of algorithms. In order to do that, go to `stream/sequence_setup/include/ConfiguredSequence.cuh` and add the algorithm to the `SEQUENCE` line as follows:

__Note: Don't forget the `#include` line__

```clike
#include "../../../cuda/test/saxpy/include/Saxpy.cuh"
...

/**
 * Especify here the algorithms to be executed in the sequence,
 * in the expected order of execution.
 */
SEQUENCE(
  ...
  prefix_sum_reduce_velo_track_hit_number_t,
  prefix_sum_single_block_velo_track_hit_number_t,
  prefix_sum_scan_velo_track_hit_number_t,
  consolidate_tracks_t,
  saxpy_t,
  ...
)
```

Keep in mind the order matters, and will define when your algorithm is scheduled. In this case, we have chosen to add it after the algorithm identified by `consolidate_tracks_t`.

Next, we need to define the arguments to be passed to our function. We need to define them in order for the dynamic scheduling machinery to properly work - that is, allocate what is needed only when it's needed, and manage the memory for us.

We will distinguish arguments just passed by value from pointers to device memory. We don't need to schedule those simply passed by value like `n` and `a`. We care however about `x` and `y`, since they require some reserving and freeing in memory.

Let's give these arguments a name that won't collide, like `dev_x` and `dev_y`. Now, we need to add them in three places. First, `arg_enum_t` in `stream/sequence_setup/include/ArgumentEnum.cuh`:

```clike
/**
 * arg_enum_t Arguments for all algorithms in the sequence.
 */
enum arg_enum_t {
  ...
  dev_velo_track_hit_number,
  dev_prefix_sum_auxiliary_array_2,
  dev_velo_track_hits,
  dev_velo_states,
  dev_x,
  dev_y
};
```

Again, order matters. Next, we will populate the arguments and their types without the `*` in `argument_tuple_t` in `stream/sequence_setup/include/SequenceSetup.cuh`:

```clike
/**
 * @brief Argument tuple definition. All arguments and their types should
 *        be populated here. The order must be the same as arg_enum_t
 *        (checked at compile time).
 */
using argument_tuple_t = std::tuple<
  ...
  Argument<arg::dev_velo_track_hit_number, uint>,
  Argument<arg::dev_prefix_sum_auxiliary_array_2, uint>,
  Argument<arg::dev_velo_track_hits, Hit>,
  Argument<arg::dev_velo_states, Velo::State>,
  Argument<arg::dev_x, float>,
  Argument<arg::dev_y, float>
>;
```

Finally, we populate the _dependency tree_, ie. where are these arguments needed. For that, go to the body of `get_sequence_dependencies`, in `stream/sequence_setup/src/SequenceSetup.cu`:

```clike
std::vector<std::vector<int>> get_sequence_dependencies() {
  ...
  sequence_dependencies[tuple_contains<saxpy_t, sequence_t>::index] = {
    arg::dev_x,
    arg::dev_y
  };
  
  return sequence_dependencies;
}
```

Optionally, we can give names to our arguments. This will help when debugging ie. the memory manager. `stream/sequence_setup/src/SequenceSetup.cu`:

```clike
std::array<std::string, std::tuple_size<argument_tuple_t>::value> get_argument_names() {
  ...
  a[arg::dev_x] = "dev_x";
  a[arg::dev_y] = "dev_y";
  return a;
}
```

Optionally (2), some types are required to live throughout the whole sequence since its creation. An argument can be specified to be persistent in memory by adding it to `SequenceSetup.cu`, function `get_sequence_output_arguments`:

```clike
std::vector<int> get_sequence_output_arguments() {
  return {
    arg::dev_atomics_storage,
    arg::dev_velo_track_hit_number,
    arg::dev_velo_track_hits
  };
}
```

### Preparing and invoking the algorithms in the sequence

Now all the pieces are in place, we are ready to prepare the algorithm and do the actual invocation.

First go to `stream/sequence/include/HostBuffers.cuh` and add the saxpy host memory pointer:

```clike
  ...
    
  // Pinned host datatypes
  uint* host_velo_tracks_atomics;
  uint* host_velo_track_hit_number;
  char* host_velo_track_hits;
  uint* host_total_number_of_velo_clusters;
  uint* host_number_of_reconstructed_velo_tracks;
  uint* host_accumulated_number_of_hits_in_velo_tracks;
  char* host_velo_states;
  uint* host_accumulated_number_of_ut_hits;

  // Saxpy
  int saxpy_N = 1<<20;
  float *host_x, *host_y;

  ...
```

Reserve that host memory in `stream/sequence/src/HostBuffers.cu`:

```clike
  ...
    
  cudaCheck(cudaMallocHost((void**)&host_velo_tracks_atomics, (2 * max_number_of_events + 1) * sizeof(int)));
  cudaCheck(cudaMallocHost((void**)&host_velo_track_hit_number, max_number_of_events * VeloTracking::max_tracks * sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_velo_track_hits, max_number_of_events * VeloTracking::max_tracks * VeloTracking::max_track_size * sizeof(Velo::Hit)));
  cudaCheck(cudaMallocHost((void**)&host_total_number_of_velo_clusters, sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_number_of_reconstructed_velo_tracks, sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_accumulated_number_of_hits_in_velo_tracks, sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_velo_states, max_number_of_events * VeloTracking::max_tracks * sizeof(Velo::State)));
  cudaCheck(cudaMallocHost((void**)&host_veloUT_tracks, max_number_of_events * VeloUTTracking::max_num_tracks * sizeof(VeloUTTracking::TrackUT)));
  cudaCheck(cudaMallocHost((void**)&host_atomics_veloUT, VeloUTTracking::num_atomics * max_number_of_events * sizeof(int)));
  cudaCheck(cudaMallocHost((void**)&host_accumulated_number_of_ut_hits, sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_accumulated_number_of_scifi_hits, sizeof(uint)));
  
  // Saxpy memory allocations
  cudaCheck(cudaMallocHost((void**)&host_x, saxpy_N * sizeof(float)));
  cudaCheck(cudaMallocHost((void**)&host_y, saxpy_N * sizeof(float)));

  ...
```

Finally, create a visitor for your newly created algorithm. Create a containing folder structure for it in `stream/sequence_visitors/test/src/`, and a new file inside named `SaxpyVisitor.cu`. Insert the following code inside:

```clike
#include "StreamVisitor.cuh"
#include "Saxpy.cuh"

template<>
void StreamVisitor::visit<saxpy_t>(
  saxpy_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
    // Saxpy test
    int saxpy_N = 1<<20;
    for (int i = 0; i < saxpy_N; i++) {
      host_buffers.host_x[i] = 1.0f;
      host_buffers.host_y[i] = 2.0f;
    }

    // Set arguments size
    arguments.set_size<arg::dev_x>(saxpy_N);
    arguments.set_size<arg::dev_y>(saxpy_N);

    // Reserve required arguments for this algorithm in the sequence
    scheduler.setup_next(arguments, sequence_step);

    // Copy memory from host to device
    cudaCheck(cudaMemcpyAsync(
      arguments.offset<arg::dev_x>(),
      host_buffers.host_x,
      saxpy_N * sizeof(float),
      cudaMemcpyHostToDevice,
      cuda_stream
    ));

    cudaCheck(cudaMemcpyAsync(
      arguments.offset<arg::dev_y>(),
      host_buffers.host_y,
      saxpy_N * sizeof(float),
      cudaMemcpyHostToDevice,
      cuda_stream
    ));

    // Setup opts for kernel call
    state.set_opts(dim3((saxpy_N+255)/256), dim3(256), cuda_stream);
    
    // Setup arguments for kernel call
    state.set_arguments(
      arguments.offset<arg::dev_x>(),
      arguments.offset<arg::dev_y>(),
      saxpy_N,
      2.0f
    );

    // Kernel call
    state.invoke();

    // Retrieve result
    cudaCheck(cudaMemcpyAsync(
      host_buffers.host_y,
      arguments.offset<arg::dev_y>(),
      arguments.size<arg::dev_y>(),
      cudaMemcpyDeviceToHost,
      cuda_stream
    ));

    // Wait to receive the result
    cudaEventRecord(cuda_generic_event, cuda_stream);
    cudaEventSynchronize(cuda_generic_event);

    // Check the output
    float maxError = 0.0f;
    for (int i=0; i<saxpy_N; i++) {
      maxError = std::max(maxError, abs(host_buffers.host_y[i]-4.0f));
    }
    info_cout << "Saxpy max error: " << maxError << std::endl << std::endl;
    
    ...
```

We can compile the code and run the program `./cu_hlt`. If everything went well, the following text should appear:

```
Saxpy max error: 0.00
```

The cool thing is your algorithm is now part of the sequence. You can see how memory is managed, taking into account your algorithm, and how it changes on every step by appending the `-p` option: `./cu_hlt -p`

```
Sequence step 13 "saxpy_t" memory segments (MiB):
dev_velo_track_hit_number (0.01), unused (0.05), dev_atomics_storage (0.00), unused (1.30), dev_velo_track_hits (0.26), dev_x (4.00), dev_y (4.00), unused (1014.39), 
Max memory required: 9.61 MiB
```

Now you are ready to take over.

Good luck!
