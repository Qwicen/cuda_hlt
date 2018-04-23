
file(GLOB x86_clustering "src/*cpp")

# to do: why do I need nvcc to compile the x86 code?
# it depends on clustering functions in cuda/velo/mask_clustering,
# we should make them __host__ and __device__ functions
cuda_add_library( libx86Clustering STATIC
  ${x86_clustering}
  )
