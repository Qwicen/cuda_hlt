Allen: Adding a new CUDA algorithm
=====================================

This tutorial will guide you through adding a new CUDA algorithm to the `Allen` project.

SAXPY
-----

Writing an algorithm in CUDA in the `Allen` project is no different than writing it on any other GPU project. The differences are in how to invoke that program, and how to setup the options, arguments, and so on.

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
include_directories(${CMAKE_SOURCE_DIR}/stream/handlers/include)

add_library(Test STATIC
  ${test_saxpy}
)
```

Our CUDA algorithm `Saxpy.cuh` and `Saxpy.cu` will be as follows. Note we need to specify the required arguments in the `ALGORITHM`, let's give the arguments names that won't collide, like `dev_x` and `dev_y`:

```clike=
#include "Handler.cuh"
#include "ArgumentsCommon.cuh"

__global__ void saxpy(float *x, float *y, int n, float a);

ALGORITHM(saxpy, saxpy_t,
  ARGUMENTS(
    dev_x,
    dev_y
))
```

```clike=
#include "Saxpy.cuh"

__global__ void saxpy(float *x, float *y, int n, float a) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}
```

The line with `ALGORITHM` encapsulates our algorithm `saxpy` into a class with name `saxpy_t`. We will use this class from now on to be able to refer to our algorithm.
Therefore, when developing algorithms for the HLT1 chain, please add the sub-detector that your algorithm belongs to in the name so that it can be easily identified within a sequence. For example: `velo_masked_clustering_t` or `ut_pre_decode_t`.

Lastly, edit `stream/CMakeLists.txt` and modify `target_link_libraries`:

```cmake
target_link_libraries(Stream Velo Test)
```

Ready to move on.

### Integrating the algorithm in the sequence

`Allen` centers around the idea of running a __sequence of algorithms__ on input events. This sequence is predefined and will always be executed in the same order.

Some events from the input will be discarded throughout the execution, and only a fraction of them will be kept for further processing. That is conceptually the idea behind the _High Level Trigger 1_ stage of LHCb, and is what is intended to achieve with this project.

Therefore, we need to add our algorithm to the sequence of algorithms. First, make the folder visible to CMake by editing the file `stream/CMakeLists.txt` and adding:

```clike
include_directories(${CMAKE_SOURCE_DIR}/cuda/test/saxpy/include)
```

Then, add the following include to `stream/setup/include/ConfiguredSequence.cuh`:

```clike
#include "Saxpy.cuh"
```

Now, we are ready to add our algorithm to a sequence. All available sequences live in the folder `configuration/sequences/`. The sequence to execute can be chosen at compile time, by appending the name of the desired sequence to the cmake call: `cmake -DSEQUENCE=DefaultSequence ..`. For now, let's just edit the `DefaultSequence`. Add the algorithm to `configuration/sequences/DefaultSequence.h` as follows:

```clike
/**
 * Specify here the algorithms to be executed in the sequence,
 * in the expected order of execution.
 */
SEQUENCE_T(
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

In the algorithm definition we used the arguments `dev_x` and `dev_y`. We need to define the arguments, to make them available to our algorithm. Let's add these types to the common arguments, in `stream/setup/include/ArgumentsCommon.cuh`:

```clike
...
ARGUMENT(dev_x, float)
ARGUMENT(dev_y, float)
```

Optionally, some types are required to live throughout the whole sequence since its creation. An argument can be specified to be persistent in memory by adding it to the `output_arguments_t` tuple, in `AlgorithmDependencies.cuh`:

```clike
/**
 * @brief Output arguments, ie. that cannot be freed.
 * @details The arguments specified in this type will
 *          be kept allocated since their first appearance
 *          until the end of the sequence.
 */
typedef std::tuple<
  dev_atomics_storage,
  dev_velo_track_hit_number,
  dev_velo_track_hits,
  dev_atomics_veloUT,
  dev_veloUT_tracks,
  dev_scifi_tracks,
  dev_n_scifi_tracks
> output_arguments_t;
```

### Preparing and invoking the algorithms in the sequence

Now all the pieces are in place, we are ready to prepare the algorithm and do the actual invocation.

First go to `stream/sequence/include/HostBuffers.cuh` and add the saxpy host memory pointer:

```clike
  ...
    
  // Pinned host datatypes
  uint* host_velo_tracks_atomics;
  uint* host_velo_track_hit_number;
  uint* host_velo_track_hits;
  uint* host_total_number_of_velo_clusters;
  uint* host_number_of_reconstructed_velo_tracks;
  uint* host_accumulated_number_of_hits_in_velo_tracks;
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
  cudaCheck(cudaMallocHost((void**)&host_veloUT_tracks, max_number_of_events * VeloUTTracking::max_num_tracks * sizeof(VeloUTTracking::TrackUT)));
  cudaCheck(cudaMallocHost((void**)&host_atomics_veloUT, VeloUTTracking::num_atomics * max_number_of_events * sizeof(int)));
  cudaCheck(cudaMallocHost((void**)&host_accumulated_number_of_ut_hits, sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_accumulated_number_of_scifi_hits, sizeof(uint)));
  
  // Saxpy memory allocations
  cudaCheck(cudaMallocHost((void**)&host_x, saxpy_N * sizeof(float)));
  cudaCheck(cudaMallocHost((void**)&host_y, saxpy_N * sizeof(float)));

  ...
```

Finally, create a visitor for your newly created algorithm. Create a containing folder structure for it in `stream/visitors/test/src/`, and a new file inside named `SaxpyVisitor.cu`. Insert the following code inside:

```clike
#include "SequenceVisitor.cuh"
#include "Saxpy.cuh"

template<>
void SequenceVisitor::set_arguments_size<saxpy_t>(
  saxpy_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  // Set arguments size
  int saxpy_N = 1<<20;
  arguments.set_size<dev_x>(saxpy_N);
  arguments.set_size<dev_y>(saxpy_N);
}

template<>
void SequenceVisitor::visit<saxpy_t>(
  saxpy_t& state,
  const saxpy_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
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

  // Copy memory from host to device
  cudaCheck(cudaMemcpyAsync(
    arguments.offset<dev_x>(),
    host_buffers.host_x,
    saxpy_N * sizeof(float),
    cudaMemcpyHostToDevice,
    cuda_stream
  ));

  cudaCheck(cudaMemcpyAsync(
    arguments.offset<dev_y>(),
    host_buffers.host_y,
    saxpy_N * sizeof(float),
    cudaMemcpyHostToDevice,
    cuda_stream
  ));

  // Setup opts for kernel call
  state.set_opts(dim3((saxpy_N+255)/256), dim3(256), cuda_stream);
  
  // Setup arguments for kernel call
  state.set_arguments(
    arguments.offset<dev_x>(),
    arguments.offset<dev_y>(),
    saxpy_N,
    2.0f
  );

  // Kernel call
  state.invoke();

  // Retrieve result
  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_y,
    arguments.offset<dev_y>(),
    arguments.size<dev_y>(),
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
}
```

As a last step, add the visitor to `stream/CMakeLists.txt`:

```clike
...
file(GLOB stream_visitors_test "visitors/test/src/*cu")
...
add_library(Stream STATIC
${stream_visitors_test}
...
```

We can compile the code and run the program `./Allen`. If everything went well, the following text should appear:

```
Saxpy max error: 0.00
```

The cool thing is your algorithm is now part of the sequence. You can see how memory is managed, taking into account your algorithm, and how it changes on every step by appending the `-p` option: `./Allen -p`

```
Sequence step 13 "saxpy_t" memory segments (MiB):
dev_velo_track_hit_number (0.01), unused (0.05), dev_atomics_storage (0.00), unused (1.30), dev_velo_track_hits (0.26), dev_x (4.00), dev_y (4.00), unused (1014.39), 
Max memory required: 9.61 MiB
```


Before placing a merge request
==============================
Before starting to edit files, please ensure that your editor produces spaces, not tabs!

Before placing a merge request, please go through the following list and check that BOTH compilation and running work after your changes:
   * Release and debug mode `cmake -DCMAKE_BUILD_TYPE=release ..` and `cmake -DCMAKE_BUILD_TYPE=debug ..`
   * Different sequences:
      * Default sequence: `cmake -DSEQUENCE=DefaultSequence ..`
      * CPU SciFi tracking sequence: `cmake -DSEQUENCE=CPUSciFi ..`
      * CPU PV finding sequence: `cmake -DSEQUENCE=CPUPVSequence ..`
  * Compilation with ROOT (if you have a ROOT installation available): `cmake -DUSE_ROOT=TRUE ..` If you don't have ROOT available, please mention this in the merge request, then we will test it.
  

Check that you can run `./Allen` after every compilation. 
  

Now you are ready to take over!

Good luck!
