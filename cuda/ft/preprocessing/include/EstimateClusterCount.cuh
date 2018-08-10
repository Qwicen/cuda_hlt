#include <inttypes.h>
__global__ void estimate_cluster_count(uint *ft_event_offsets,  uint *dev_ft_cluster_count, char *ft_events);

struct FTRawEvent {
  uint32_t number_of_raw_banks;
  uint32_t* raw_bank_offset;
  uint32_t version;
  char* payload;

  __device__ __host__ FTRawEvent(
    const char* event
  );
};

struct FTRawBank {
  uint32_t sourceID;
  uint32_t length;
  uint32_t* data;

  __device__ __host__ FTRawBank(
    const char* raw_bank,
    unsigned int length
  );
};
