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

> Note: You may notice I changed the order of the parameters, and put first the pointers `float* x, float* y`. Putting them at the end leads to the following compiler error, which I still have to understand `
Error: Internal Compiler Error (codegen): "there was an error in verifying the lgenfe output!"
`.


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

cuda_add_library(Test STATIC
  ${test_saxpy}
)
```

Our CUDA algorithm `Saxpy.cuh` and `Saxpy.cu` will be as follows:

```clike=
__global__ void saxpy(float *x, float *y, int n, float a);
```

```clike=
#include "Saxpy.cuh"

__global__ void saxpy(float *x, float *y, int n, float a) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}
```

Lastly, edit `stream/CMakeLists.txt` and modify `target_link_libraries`:

```cmake
target_link_libraries(Stream Velo Test)
```

Ready to move on.

### Integrating the algorithm in the sequence

`cuda_hlt` centers around the idea of running a __sequence of algorithms__ on input events. This sequence is predefined and will always be executed in the same order.

Some events from the input will be discarded throughout the execution, and only a fraction of them will be kept for further processing. That is conceptually the idea behind the _High Level Trigger 1_ stage of LHCb, and is what is intended to achieve with this project.

Therefore, we need to add our algorithm to the sequence of algorithms. In order to do that, go to `stream/sequence_setup/include/SequenceArgumentEnum.cuh` and add the algorithm to the `enum seq_enum_t` type as follows:

```clike
/**
 * seq_enum_t contains all steps of the sequence in the expected
 *            order of execution.
 */
enum seq_enum_t {
  ...
  prefix_sum_single_block_velo_track_hit_number,
  prefix_sum_scan_velo_track_hit_number,
  consolidate_tracks,
  saxpy
};
```

Keep in mind the order matters, and will define when your algorithm is scheduled. In this case, we have chosen to add it after the algorithm identified by `consolidate_tracks`. Next, we need to add the __function identifier__ to the algorithms tuple. Our function identifier (the name of the function) is __saxpy__. Go to `stream/sequence_setup/include/SequenceSetup.cuh`:

__Note: Don't forget the `#include` line__

```clike
#include "../../../cuda/test/saxpy/include/Saxpy.cuh"
...

/**
 * @brief Algorithm tuple definition. All algorithms in the sequence
 *        should be added here in the same order as seq_enum_t
 *        (this condition is checked at compile time).
 */
constexpr auto sequence_algorithms() {
  return std::make_tuple(
    ...
    prefix_sum_single_block,
    prefix_sum_scan,
    consolidate_tracks,
    saxpy
  );
}
```

Next, we need to define the arguments to be passed to our function. We need to define them in order for the dynamic scheduling machinery to properly work - that is, allocate what is needed only when it's needed, and manage the memory for us.

We will distinguish arguments just passed by value from pointers to device memory. We don't need to schedule those simply passed by value like `n` and `a`. We care however about `x` and `y`, since they require some reserving and freeing in memory.

Let's give these arguments a name that won't collide, like `dev_x` and `dev_y`. Now, we need to add them in three places. First, `arg_enum_t` in `stream/sequence_setup/include/SequenceArgumentEnum.cuh`:

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
  sequence_dependencies[seq::saxpy] = {
    arg::dev_x,
    arg::dev_y
  };
  
  return sequence_dependencies;
}
```

Optionally, we can give names to our algorithm and arguments. This will help when debugging ie. the memory manager. `stream/sequence_setup/src/SequenceSetup.cu`:

```clike
std::array<std::string, std::tuple_size<algorithm_tuple_t>::value> get_sequence_names() {
  ...
  a[seq::saxpy] = "Saxpy test";
  return a;
}

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

First go to `stream/sequence/include/Stream.cuh` and add the saxpy host memory pointer:

```clike
  ...
    
  // Pinned host datatypes
  int* host_number_of_tracks;
  int* host_accumulated_tracks;
  uint* host_velo_track_hit_number;
  Hit* host_velo_track_hits;
  uint* host_total_number_of_velo_clusters;
  uint* host_number_of_reconstructed_velo_tracks;
  uint* host_accumulated_number_of_hits_in_velo_tracks;

  // Saxpy
  int saxpy_N = 1<<20;
  float *host_x, *host_y;

  ...
```

Reserve that host memory in `stream/sequence/src/Stream.cu`:

```clike
  ...
    
  // Memory allocations for host memory (copy back)
  cudaCheck(cudaMallocHost((void**)&host_number_of_tracks, max_number_of_events * sizeof(int)));
  cudaCheck(cudaMallocHost((void**)&host_accumulated_tracks, max_number_of_events * sizeof(int)));
  cudaCheck(cudaMallocHost((void**)&host_velo_track_hit_number, max_number_of_events * VeloTracking::max_tracks * sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_velo_track_hits, max_number_of_events * VeloTracking::max_tracks * 20 * sizeof(Hit)));
  cudaCheck(cudaMallocHost((void**)&host_total_number_of_velo_clusters, sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_number_of_reconstructed_velo_tracks, sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_accumulated_number_of_hits_in_velo_tracks, sizeof(uint)));
  
  // Saxpy memory allocations
  cudaCheck(cudaMallocHost((void**)&host_x, saxpy_N * sizeof(float)));
  cudaCheck(cudaMallocHost((void**)&host_y, saxpy_N * sizeof(float)));

  ...
```

Finally, go to `stream/sequence/src/StreamSequence.cu` and insert the following code after _Consolidate tracks_:


```clike
    ...
    
    // Consolidate tracks
    argument_sizes[arg::dev_velo_track_hits] = argen.size<arg::dev_velo_track_hits>(host_accumulated_number_of_hits_in_velo_tracks[0]);
    argument_sizes[arg::dev_velo_states] = argen.size<arg::dev_velo_states>(host_number_of_reconstructed_velo_tracks[0]);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::consolidate_tracks>().set_opts(dim3(number_of_events), dim3(32), stream);
    sequence.item<seq::consolidate_tracks>().set_arguments(
      argen.generate<arg::dev_atomics_storage>(argument_offsets),
      argen.generate<arg::dev_tracks>(argument_offsets),
      argen.generate<arg::dev_velo_track_hit_number>(argument_offsets),
      argen.generate<arg::dev_velo_cluster_container>(argument_offsets),
      argen.generate<arg::dev_estimated_input_size>(argument_offsets),
      argen.generate<arg::dev_module_cluster_num>(argument_offsets),
      argen.generate<arg::dev_velo_track_hits>(argument_offsets),
      argen.generate<arg::dev_velo_states>(argument_offsets)
    );
    sequence.item<seq::consolidate_tracks>().invoke();

    // Saxpy test
    for (int i = 0; i < saxpy_N; i++) {
      host_x[i] = 1.0f;
      host_y[i] = 2.0f;
    }

    // Set arguments size
    argument_sizes[arg::dev_x] = argen.size<arg::dev_x>(saxpy_N);
    argument_sizes[arg::dev_y] = argen.size<arg::dev_y>(saxpy_N);

    // Reserve required arguments for this algorithm in the sequence
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);

    // Copy memory from host to device
    cudaCheck(cudaMemcpyAsync(
      argen.generate<arg::dev_x>(argument_offsets),
      host_x,
      saxpy_N * sizeof(float),
      cudaMemcpyHostToDevice,
      stream
    ));
    cudaCheck(cudaMemcpyAsync(
      argen.generate<arg::dev_y>(argument_offsets),
      host_y,
      saxpy_N * sizeof(float),
      cudaMemcpyHostToDevice,
      stream
    ));

    // Setup opts for kernel call
    sequence.item<seq::saxpy>().set_opts(dim3((saxpy_N+255)/256), dim3(256), stream);
    
    // Setup arguments for kernel call
    sequence.item<seq::saxpy>().set_arguments(
      argen.generate<arg::dev_x>(argument_offsets),
      argen.generate<arg::dev_y>(argument_offsets),
      saxpy_N,
      2.0f
    );

    // Kernel call
    sequence.item<seq::saxpy>().invoke();

    // Retrieve result
    cudaCheck(cudaMemcpyAsync(host_y,
      argen.generate<arg::dev_y>(argument_offsets),
      argen.size<arg::dev_y>(saxpy_N),
      cudaMemcpyDeviceToHost,
      stream
    ));

    // Wait to receive the result
    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);

    // Check the output
    float maxError = 0.0f;
    for (int i = 0; i < saxpy_N; i++) {
      maxError = std::max(maxError, abs(host_y[i]-4.0f));
    }
    info_cout << "Saxpy max error: " << maxError << std::endl << std::endl;
    
    ...
```

We can compile the code and run the program with simple settings, something like `./cu_hlt -f ../input/minbias/velopix_raw -e ../input/minbias/ut_hits -g ../input/geometry/`. If everything went well, the following text should appear:

```
Saxpy max error: 0.00
```

The cool thing is your algorithm is now part of the sequence. You can see how memory is managed, taking into account your algorithm, and how it changes on every step by appending the `-p` option: `./cu_hlt -f ../input/minbias/velopix_raw -e ../input/minbias/ut_hits -g ../input/geometry/ -p`

```
Sequence step 13 "Saxpy test" memory segments (MiB):
dev_velo_track_hit_number (0.01), unused (0.05), dev_atomics_storage (0.00), unused (1.30), dev_velo_track_hits (0.26), dev_x (4.00), dev_y (4.00), unused (1014.39), 
Max memory required: 9.61 MiB
```

Now you are ready to take over.

Good luck!

### Bonus: Extending a Handler

Handlers are used internally to deal with each algorithm in the sequence. A Handler deduces the argument types from the kernel function identifier, and exposes the `set_opts` and `set_arguments` methods.

Handlers can be specialized and extended for a particular algorithm. This can be useful under certain situations, ie. if one wants to develop a checker method, or a printout method.

Coming back to Saxpy, we may not want to have the checking code laying around in the `run_sequence` body. Particularly, this code could live somewhere else:

```clike
    // Retrieve result
    cudaCheck(cudaMemcpyAsync(host_y,
      argen.generate<arg::dev_y>(argument_offsets),
      argen.size<arg::dev_y>(saxpy_N),
      cudaMemcpyDeviceToHost,
      stream
    ));

    // Wait to receive the result
    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);

    // Check the output
    float maxError = 0.0f;
    for (int i = 0; i < saxpy_N; i++) {
      maxError = std::max(maxError, abs(host_y[i]-4.0f));
    }
    info_cout << "Saxpy max error: " << maxError << std::endl << std::endl;
```

Let's start by specializing a Handler and registering that specialization. Create a new file in `stream/handlers/include/` and name it `HandlerSaxpy.cuh`. These are the contents:

```clike=
#pragma once

#include "../../../main/include/CudaCommon.h"
#include "../../../main/include/Logger.h"
#include "../../sequence_setup/include/SequenceArgumentEnum.cuh"
#include "HandlerDispatcher.cuh"
#include <iostream>

template<typename R, typename... T>
struct HandlerSaxpy : public Handler<seq::saxpy, R, T...> {
  HandlerSaxpy() = default;
  HandlerSaxpy(R(*param_function)(T...))
  : Handler<seq::saxpy, R, T...>(param_function) {}

  // Add your own methods
};

// Register partial specialization
template<>
struct HandlerDispatcher<seq::saxpy> {
  template<typename R, typename... T>
  using H = HandlerSaxpy<R, T...>;
};

```

Next, register that Handler. Modify `HandlerMaker.cuh` and add the Handler we just created:

```clike
// Note: Add here additional custom handlers
#include "HandlerSaxpy.cuh"
```

Now we are ready to extend HandlerSaxpy. The way this works is by using a partial specialization of `HandlerDispatcher` and defining `H` as our specific Handler. `HandlerMaker` takes care of the rest.

We can now add a `check` method to `HandlerSaxpy`:

```clike
  // Add your own methods
  void check(
    float* host_y,
    int saxpy_N,
    float* dev_y,
    size_t dev_y_size,
    cudaStream_t& stream,
    cudaEvent_t& cuda_generic_event
  ) {
    // Retrieve result
    cudaCheck(cudaMemcpyAsync(host_y,
      dev_y,
      dev_y_size,
      cudaMemcpyDeviceToHost,
      stream
    ));

    // Wait to receive the result
    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);

    // Check the output
    float maxError = 0.0f;
    for (int i = 0; i < saxpy_N; i++) {
      maxError = std::max(maxError, abs(host_y[i]-4.0f));
    }
    info_cout << "Saxpy max error: " << maxError << std::endl << std::endl;
  }
```

And refactor `StreamSequence.cu` to reflect this change:

```clike
    // Kernel call
    sequence.item<seq::saxpy>().invoke();

    // Check result
    sequence.item<seq::saxpy>().check(
      host_y,
      saxpy_N,
      argen.generate<arg::dev_y>(argument_offsets),
      argen.size<arg::dev_y>(saxpy_N),
      stream,
      cuda_generic_event
    );
```

Now you are a `cuda_hlt` hacker.

> Note: You can find the full example under the branch `saxpy_test`.
